#!/usr/bin/env python
"""Deployment: roll candidates through T_theta, score with p_succ, execute argmax.

    python scripts/run_v126_wm_select.py --wm <T_theta.pt> --head <p_succ.pt> \
        --arm wm --n 8 --panel 48

GOAL ANCHOR (CLAUDE.md). This is the whole point: framework 5.2 used at
deployment. At every chunk boundary the frozen policy proposes n candidates; each
is rolled forward in LATENT space, z~ = T_theta(z_t, E_a(u^i)); D_theta's p_succ
head scores the PREDICTED latent; the argmax is executed. Policy weights frozen,
prompt unchanged, no reconstruction anywhere.

THREE ARMS, registered before running. The middle one is the control that decides
whether the world model contributes anything:

  base    stock policy, N=1                    the deployment baseline
  random  sample n candidates, execute a RANDOM one
  wm      sample n candidates, execute argmax p_succ(T_theta(z, u))

`wm` MUST beat `random` for the world model to be doing work. `wm` beating `base`
alone proves nothing: drawing n samples and taking any of them already changes the
action distribution. The 2026-08-27 session's error was claiming a mechanism
without the arm that could refute it; that arm is built in here from the start.

chain1b was chosen because v123 measured within-state outcome variance of 0.119
(within fraction 0.517) - there is genuinely something to select between. On
chain2b that quantity is zero and every arm would be identical by construction.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.v080_bench import episode_length, make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402
from lcwm.v08r_contract import clopper_pearson_lower, clopper_pearson_upper  # noqa: E402
from train_v122_latent_wm import Transition  # noqa: E402
from train_v125_psucc_head import PSucc  # noqa: E402

C = 10
OUT = REPO / "results" / "v126_wm_select"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--arm", choices=["base", "random", "wm"], required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="candidate sampling noise; 0 = the policy's own ODE")
    ap.add_argument("--depth", type=int, default=1,
                    help="how many times T_theta is composed before scoring. v127: "
                         "score range grows 0.0026 -> 0.0095 with depth at sigma=0, "
                         "and 0.0145 -> 0.0415 at sigma=3")
    ap.add_argument("--head-kind", choices=["p_succ", "delta_w"], default="p_succ")
    ap.add_argument("--panel", type=int, default=48)
    ap.add_argument("--panel-start", type=int, default=6500)
    a = ap.parse_args()
    L = episode_length(a.task)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    tag = ("base" if a.arm == "base" else
           f"{a.arm}_n{a.n}_s{a.sigma}_d{a.depth}_{a.head_kind}_logit")
    out = OUT / f"{a.task}_{tag}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.wm, weights_only=False)
    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    mu, sd = ck["mu"], ck["sd"]
    hk = torch.load(a.head, weights_only=False)
    head = PSucc(hk["zdim"]); head.load_state_dict(hk["state_dict"]); head.eval()
    print(f"arm={a.arm} n={a.n} sigma={a.sigma} task={a.task} L={L}")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    env = make_v080_env(a.task)
    PANEL = tuple(range(a.panel_start, a.panel_start + a.panel))

    rows, steps = [], 0
    for seed in PANEL:
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        atoms = goal_atoms(env)
        t, done, succ, picks = 0, False, None, []
        while not done and t < L:
            if a.arm == "base":
                act = runner.select_action(obs, env.task_description)
                obs, _r, tm, tr, inf = env.step(act); t += 1
                done = bool(tm or tr)
                if succ is None and bool(inf.get("is_success", False)):
                    succ = t
                if (t % 10 == 0 or done) and succ is None and predicate_bits(env, atoms).all():
                    succ = t
                continue
            po = runner._obs_to_policy_batch(obs, env.task_description)
            with torch.no_grad():
                pf = prefix_forward(runner.policy, po)
                cand = sample_chunks(runner.policy, po, a.n, seed=int(seed) * 7919 + t,
                                     prefix=pf, sigma=a.sigma)[:, :C].detach().float().cpu()
                z = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                       pf.pad_masks[0].detach().cpu())
                zt = ((z - mu) / sd).unsqueeze(0).expand(a.n, -1)
                for _ in range(a.depth):                  # roll the LATENT forward
                    zt = T(zt, cand)
                # Rank on LOGITS. v130: after sigmoid the eight candidates collapse
                # to a single float (#distinct 1.0 of 8) while the logits stay fully
                # distinct (8.0 of 8, sd 0.078-0.760). Ranking on the probability
                # destroyed the ordering numerically, which is what the v126/v129
                # nulls actually measured.
                sc = head(zt)                             # head on the PREDICTED latent
            del pf
            k = int(rng.integers(a.n)) if a.arm == "random" else int(torch.argmax(sc))
            picks.append({"t": t, "k": k, "logit": float(sc[k]),
                          "min": float(sc.min()), "max": float(sc.max()),
                          "n_distinct": int(len(set(round(float(v), 12) for v in sc)))})
            for act in runner.chunk_to_env(cand[k]):
                obs, _r, tm, tr, inf = env.step(act); t += 1
                done = bool(tm or tr)
                if succ is None and bool(inf.get("is_success", False)):
                    succ = t
                if done: break
            if succ is None and predicate_bits(env, atoms).all():
                succ = t
        steps += t
        rows.append({"seed": seed, "success": succ is not None, "success_step": succ,
                     "steps": t, "picks": picks})
        print(f"{tag} s{seed}: succ={succ is not None!s:5s} t={t} ({steps} steps)",
              flush=True)

    k = sum(r["success"] for r in rows)
    summary = {"task": a.task, "arm": a.arm, "tag": tag, "utc": stamp,
               "n_candidates": a.n, "sigma": a.sigma, "depth": a.depth,
               "head_kind": a.head_kind, "wm": str(a.wm),
               "head": str(a.head), "panel": [PANEL[0], PANEL[-1], len(PANEL)],
               "successes": k, "n": len(rows), "rate": k / len(rows),
               "cp95": [clopper_pearson_lower(k, len(rows)),
                        clopper_pearson_upper(k, len(rows))],
               "env_steps": steps, "episodes": rows,
               "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{tag}: {k}/{len(rows)} = {k/len(rows):.3f} "
          f"CP95 [{summary['cp95'][0]:.3f},{summary['cp95'][1]:.3f}] "
          f"{steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
