#!/usr/bin/env python
"""Deployment: branch at a few points, rank with T_theta + D_theta, execute argmax.

    python scripts/run_v141_wm_branch_deploy.py --wm-proprio <T_theta.pt> \
        --head <rank_ensemble.pt> --arm wm --panel 64 --panel-start 6800

GOAL ANCHOR (CLAUDE.md). All four axes:
  task    chain1b_lr2, pi_0 low success
  object  candidates rolled forward by T_theta in LATENT space; D_theta scores the
          PREDICTED latent. v139: this ranks better than feeding the raw action
          (+0.024 AUC, sign p=0.041, CI excludes 0)
  target  latent only, no reconstruction anywhere
  data    T_theta from deployment rollouts; D_theta from deployment branch labels

Every constant is forced by a measurement, not chosen:
  branch at {0,40}  v132: sigma injected ONCE gives random-pick 0.562 vs 0.469 at
                    sigma=0, but injecting at all ~25 boundaries scored 0.354.
                    Diversity belongs at a few branch points.
  sigma 3           v132 ceiling 0.875 vs 0.688 at sigma=0
  depth 4           v138: best or tied-best ranking at every label fraction
  ensemble          single fits have flipped conclusions here three times

ARMS, registered before running. `wm` must beat `random` at the SAME branch points
and sigma; beating `base` alone would only show that branching changes behaviour.
Panel 6800-6863 is disjoint from every seed range used to fit anything.
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
from train_v137_rank_head import Head  # noqa: E402
from collect_v121_deploy_latents import proprio  # noqa: E402

C = 10
OUT = REPO / "results" / "v141_branch_deploy"

FIT_RANGES = [(6000, 6064), (6200, 6216), (6300, 6396), (6500, 6548), (6700, 6764)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm-proprio", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--arm", choices=["base", "random", "wm"], required=True)
    ap.add_argument("--score-mode", choices=["wm", "pre_enc", "displacement"],
                    default="wm",
                    help="ATTRIBUTION. 'wm' rolls the latent forward with T_theta. "
                         "'pre_enc' uses T_theta's FROZEN pretrained action encoder "
                         "but never applies the transition - the ablation that "
                         "isolates the rollout, because v137's `direct` arm used a "
                         "RANDOM-INIT encoder and so controlled for pretraining, not "
                         "for rolling forward. 'displacement' is a zero-parameter "
                         "score ||sum_t u_t||, which already reaches within-group "
                         "AUC 0.585 offline.")
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--branch-at", type=int, nargs="+", default=[0, 40])
    ap.add_argument("--panel", type=int, default=64)
    ap.add_argument("--panel-start", type=int, default=6800)
    a = ap.parse_args()
    L = episode_length(a.task)
    PANEL = tuple(range(a.panel_start, a.panel_start + a.panel))
    for lo, hi in FIT_RANGES:
        assert not (set(PANEL) & set(range(lo, hi))), \
            f"panel overlaps a range used for fitting: {lo}-{hi}"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    tag = (a.arm if a.arm == "base" else
           f"{a.arm}_n{a.n}_s{a.sigma}" +
           ("" if a.score_mode == "wm" else f"_{a.score_mode}"))
    out = OUT / f"{a.task}_{tag}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.wm_proprio, weights_only=False)
    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    hk = torch.load(a.head, weights_only=False)
    ens = []
    for st in hk["states"]:
        m = Head(hk["zdim"], hk["c"], hk["adim"], mode="wm")
        m.load_state_dict(st); m.eval(); ens.append(m)
    mu_h, sd_h, mu_p, sd_p, depth = (hk["mu_h"], hk["sd_h"], hk["mu_p"],
                                     hk["sd_p"], hk["depth"])
    print(f"arm={a.arm} n={a.n} sigma={a.sigma} branch_at={a.branch_at} "
          f"depth={depth} ensemble={len(ens)} panel={PANEL[0]}-{PANEL[-1]}")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    env = make_v080_env(a.task)

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
            if a.arm != "base" and t in a.branch_at:
                po = runner._obs_to_policy_batch(obs, env.task_description)
                with torch.no_grad():
                    pf = prefix_forward(runner.policy, po)
                    cand = sample_chunks(runner.policy, po, a.n,
                                         seed=int(seed) * 7919 + t, prefix=pf,
                                         sigma=a.sigma)[:, :C].detach().float().cpu()
                    h = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                           pf.pad_masks[0].detach().cpu())
                    z0 = ((proprio(obs) - mu_p) / sd_p).unsqueeze(0).expand(a.n, -1)
                    Hn = ((h - mu_h) / sd_h).unsqueeze(0).expand(a.n, -1)
                    if a.score_mode == "displacement":
                        sc = cand.sum(1).norm(dim=-1)      # zero-parameter baseline
                    elif a.score_mode == "pre_enc":
                        X = torch.cat([Hn, z0, T.enc(cand)], -1)   # frozen E_a, NO rollout
                        sc = torch.stack([m(X) for m in ens]).mean(0)
                    else:
                        zt = z0
                        for _ in range(depth):
                            zt = T(zt, cand)  # roll the LATENT forward
                        sc = torch.stack([m(torch.cat([Hn, zt], -1)) for m in ens]).mean(0)
                del pf
                k = int(rng.integers(a.n)) if a.arm == "random" else int(torch.argmax(sc))
                picks.append({"t": t, "k": k, "logit": float(sc[k]),
                              "min": float(sc.min()), "max": float(sc.max())})
                for act in runner.chunk_to_env(cand[k]):
                    if done: break
                    obs, _r, tm, tr, inf = env.step(act); t += 1
                    done = bool(tm or tr)
                    if succ is None and bool(inf.get("is_success", False)):
                        succ = t
                runner.reset()
                if succ is None and predicate_bits(env, atoms).all():
                    succ = t
                continue
            obs, _r, tm, tr, inf = env.step(runner.select_action(obs, env.task_description))
            t += 1; done = bool(tm or tr)
            if succ is None and bool(inf.get("is_success", False)):
                succ = t
            if (t % 10 == 0 or done) and succ is None and predicate_bits(env, atoms).all():
                succ = t
        steps += t
        rows.append({"seed": seed, "success": succ is not None, "success_step": succ,
                     "steps": t, "picks": picks})
        print(f"{tag} s{seed}: succ={succ is not None!s:5s} t={t} ({steps} steps)",
              flush=True)

    k = sum(r["success"] for r in rows)
    summary = {"task": a.task, "arm": a.arm, "tag": tag, "utc": stamp,
               "n_candidates": a.n, "sigma": a.sigma, "branch_at": a.branch_at,
               "depth": depth, "score_mode": a.score_mode,
               "ensemble": len(ens), "wm": str(a.wm_proprio),
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
