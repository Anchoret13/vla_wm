#!/usr/bin/env python
"""Deploy VLA + residual trained in imagination. The only measurement that counts.

    python scripts/run_v163_residual_deploy.py --wm <wm.pt> --actor <actor.pt> \
        --arm residual --panel 96 --panel-start 7500

GOAL ANCHOR (CLAUDE.md). The world model is used here for POLICY IMPROVEMENT, not
ranking: a bounded residual Delta_phi(z) trained purely on imagined latent rollouts
is added to the VLA's own chunk. The VLA is the prior and the trust region; its
weights and prompt are untouched.

    executed = chunk_to_env(u_VLA + Delta_phi(z)),  ||Delta|| <= scale

ARMS: base (VLA alone), residual (Delta trained WITH imagination), no_imag (the
same Delta trained against the same value on real next states only, never rolling
the model forward). `residual` must beat `no_imag` for the transition to be what
contributed - beating base alone would only show that some residual helps.

What is already known and bounds the expectation: the two actors' residuals have
cosine similarity +0.79 and identical imagined gains (+0.0141 vs +0.0142), and both
saturate 96% of the norm bound - the optimiser is pinned to the constraint, which
is what value extrapolation looks like. The imagined gain is therefore not evidence;
this run is.
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
from lcwm.sampler import prefix_forward  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.v080_bench import episode_length, make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402
from lcwm.v08r_contract import clopper_pearson_lower, clopper_pearson_upper  # noqa: E402
from train_v153_td_value_wm import ValueWM, load_valuewm  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402
from collect_v121_deploy_latents import proprio  # noqa: E402

C = 10
OUT = REPO / "results" / "v163_residual_deploy"
FIT_RANGES = [(6000, 6064), (6200, 6216), (6300, 6396), (6500, 6548),
              (6700, 6764), (6800, 6864), (6900, 7028), (7100, 7228), (7300, 7396)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--actor", type=Path, default=None)
    ap.add_argument("--arm", choices=["base", "residual"], required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--window", type=int, default=0,
                    help="0 = apply for the whole episode")
    ap.add_argument("--panel", type=int, default=96)
    ap.add_argument("--panel-start", type=int, default=7500)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    L = episode_length(a.task)
    PANEL = tuple(range(a.panel_start, a.panel_start + a.panel))
    for lo, hi in FIT_RANGES:
        assert not (set(PANEL) & set(range(lo, hi))), f"panel overlaps fit range {lo}-{hi}"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    tag = a.tag or a.arm
    out = OUT / f"{a.task}_{tag}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.wm, weights_only=False)
    wm = load_valuewm(ck, ck["obs_dim"], ck["c"], ck["adim"], ck["zdim"],
                 encoder=ck.get("encoder", "mlp"))
    wm.eval()
    actor = None
    if a.arm == "residual":
        akt = torch.load(a.actor, weights_only=False)
        actor = ResidualActor(akt["zdim"], akt["c"], akt["adim"], scale=akt["scale"])
        actor.load_state_dict(akt["state_dict"]); actor.eval()
        print(f"actor scale {akt['scale']} from {akt['arm']}")
    mu_o, sd_o = ck["mu_o"], ck["sd_o"]
    print(f"arm={a.arm} tag={tag} panel {PANEL[0]}-{PANEL[-1]} window={a.window or 'all'}")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    env = make_v080_env(a.task)

    rows, steps = [], 0
    for seed in PANEL:
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        atoms = goal_atoms(env)
        t, done, succ, dmag = 0, False, None, []
        while not done and t < L:
            if actor is not None and (a.window == 0 or t < a.window):
                po = runner._obs_to_policy_batch(obs, env.task_description)
                with torch.no_grad():
                    pf = prefix_forward(runner.policy, po)
                    h = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                           pf.pad_masks[0].detach().cpu())
                    chunk = runner.sample_chunk(obs, env.task_description
                                                )[0, :C].detach().float().cpu()
                del pf
                o = torch.cat([h, proprio(obs)])
                with torch.no_grad():
                    z = wm.encode(((o - mu_o) / sd_o).unsqueeze(0))
                    d = actor(z)[0]
                dmag.append(float(d.abs().mean()))
                for act in runner.chunk_to_env(chunk + d):
                    if done: break
                    obs, _r, tm, tr, inf = env.step(act); t += 1
                    done = bool(tm or tr)
                    if succ is None and bool(inf.get("is_success", False)):
                        succ = t
                runner.reset()
                if succ is None and predicate_bits(env, atoms).all():
                    succ = t
                continue
            obs, _r, tm, tr, inf = env.step(
                runner.select_action(obs, env.task_description))
            t += 1; done = bool(tm or tr)
            if succ is None and bool(inf.get("is_success", False)):
                succ = t
            if (t % 10 == 0 or done) and succ is None and predicate_bits(env, atoms).all():
                succ = t
        steps += t
        rows.append({"seed": seed, "success": succ is not None, "success_step": succ,
                     "steps": t, "mean_abs_delta": float(np.mean(dmag)) if dmag else 0.0})
        print(f"{tag} s{seed}: succ={succ is not None!s:5s} t={t} ({steps} steps)",
              flush=True)

    k = sum(r["success"] for r in rows)
    summary = {"task": a.task, "arm": a.arm, "tag": tag, "utc": stamp,
               "wm": str(a.wm), "actor": str(a.actor) if a.actor else None,
               "window": a.window, "panel": [PANEL[0], PANEL[-1], len(PANEL)],
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
