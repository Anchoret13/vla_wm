#!/usr/bin/env python
"""Deploy frozen pi0.5 + a residual trained by backprop through the belief model.

    python scripts/run_v206_belief_residual_deploy.py --actor <actor.pt> \
        --panel 96 --panel-start 7600 --tag wmres

GOAL ANCHOR (CLAUDE.md, five axes).
  task       chain1b_lr2, frozen pi0.5 at 45/96 = 0.469
  object     T_th(b_t, E_a(u)) rolled forward under the CANDIDATE action
  target     future LATENT e^_{t+1}; no pixels, no reconstruction anywhere
  data       deployment tapes only
  placement  the belief conditions the action; the residual improves the policy
             THROUGH the model - the half RB-VLA does not have

The belief is recurrent and carried across the episode, and it is fed the action
that was ACTUALLY executed (chunk + Delta) at the previous step, matching the
causal convention it was trained under: b_t depends on u_{t-1}, never on u_t.

WHAT WOULD OVERTURN A POSITIVE READING, both pre-registered:
  the --no-action control actor (its transition ignores the action) scoring the
  same, and the model-free AWR arm at 63/96 = 0.656 not being beaten.
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
from train_v157_residual_actor import ResidualActor  # noqa: E402
from train_v205_action_belief import ActionBelief  # noqa: E402
from collect_v121_deploy_latents import proprio  # noqa: E402

C = 10
OUT = REPO / "results" / "v206_belief_residual"
FIT_RANGES = [(6000, 6064), (6200, 6216), (6300, 6396), (6500, 6548),
              (6700, 6764), (6800, 6864), (6900, 7028), (7100, 7228), (7300, 7396)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", type=Path, required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--panel", type=int, default=96)
    ap.add_argument("--panel-start", type=int, default=7600)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    L = episode_length(a.task)
    PANEL = tuple(range(a.panel_start, a.panel_start + a.panel))
    for lo, hi in FIT_RANGES:
        assert not (set(PANEL) & set(range(lo, hi))), f"panel overlaps fit {lo}-{hi}"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    ck = torch.load(a.actor, weights_only=False)
    tag = a.tag or ck["arm"]
    out = OUT / f"{a.task}_{tag}_{stamp}"; out.mkdir(parents=True, exist_ok=True)
    zdim, c_, adim = ck["dims"]
    m = ActionBelief(zdim, c_, adim, use_action=ck["use_action"])
    m.load_state_dict(ck["model"]); m.eval()
    actor = ResidualActor(ck["zdim"], c_, adim, scale=ck["scale"])
    actor.load_state_dict(ck["state_dict"]); actor.eval()
    mu, sd, bmu, bsd = ck["mu"], ck["sd"], ck["bmu"], ck["bsd"]
    print(f"arm={ck['arm']} action-conditioned={ck['use_action']} "
          f"scale={ck['scale']} panel {PANEL[0]}-{PANEL[-1]}")

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
        b = torch.zeros(1, m.bdim)           # belief resets each episode
        u_prev = torch.zeros(1, c_, adim)    # no action precedes the first chunk
        while not done and t < L:
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
                e = m.enc(((o - mu) / sd).unsqueeze(0))
                b = m.step(b, e, u_prev)              # CAUSAL: uses u_{t-1}
                d = actor(torch.cat([e, (b - bmu) / bsd], -1))
            executed = chunk.unsqueeze(0) + d
            dmag.append(float(d.abs().mean()))
            u_prev = executed.detach()
            for act in runner.chunk_to_env(executed[0]):
                if done:
                    break
                obs, _r, tm, tr, inf = env.step(act); t += 1
                done = bool(tm or tr)
                if succ is None and bool(inf.get("is_success", False)):
                    succ = t
            runner.reset()
            if succ is None and predicate_bits(env, atoms).all():
                succ = t
        steps += t
        rows.append({"seed": seed, "success": succ is not None, "success_step": succ,
                     "steps": t, "mean_abs_delta": float(np.mean(dmag)) if dmag else 0.0})
        print(f"{tag} s{seed}: succ={succ is not None!s:5s} t={t} ({steps} steps)",
              flush=True)

    k = sum(r["success"] for r in rows)
    summary = {"task": a.task, "tag": tag, "utc": stamp, "actor": str(a.actor),
               "use_action": ck["use_action"], "scale": ck["scale"],
               "panel": [PANEL[0], PANEL[-1], len(PANEL)],
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
