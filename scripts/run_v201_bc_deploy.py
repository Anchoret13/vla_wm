#!/usr/bin/env python
"""Deploy a from-scratch BC policy, with or without belief conditioning.

RB-VLA's core ablation. The trained policy REPLACES pi-0.5's action output - this is
the first thing in this project that is not a frozen policy plus a correction.
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
from train_v190_belief_ceiling import Belief  # noqa: E402
from train_v200_rbvla_repro import BCPolicy  # noqa: E402
from collect_v121_deploy_latents import proprio  # noqa: E402

C = 10
OUT = REPO / "results" / "v201_bc_deploy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--panel", type=int, default=96)
    ap.add_argument("--panel-start", type=int, default=7600)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    ck = torch.load(a.policy, weights_only=False)
    pol = BCPolicy(ck["indim"], ck["c"], ck["adim"])
    pol.load_state_dict(ck["state_dict"]); pol.eval()
    bel = None
    if ck.get("belief") is not None:
        zd, c_, ad = ck["belief_dims"]
        bel = Belief(zd, c_, ad); bel.load_state_dict(ck["belief"]); bel.eval()
    L = episode_length(a.task)
    tag = a.tag or f"bc_{ck['condition']}"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.task}_{tag}_{stamp}"; out.mkdir(parents=True, exist_ok=True)
    print(f"BC policy, conditioning '{ck['condition']}', {ck['indim']} dims")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    env = make_v080_env(a.task)
    PANEL = tuple(range(a.panel_start, a.panel_start + a.panel))

    rows, steps = [], 0
    for seed in PANEL:
        torch.manual_seed(seed); np.random.seed(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        atoms = goal_atoms(env)
        # last_u is read only under `bstate is not None`, which cannot happen
        # before the first assignment below - but binding it here is what makes
        # that argument checkable instead of a promise (F821).
        t, done, succ, bstate, last_u = 0, False, None, None, None
        while not done and t < L:
            po = runner._obs_to_policy_batch(obs, env.task_description)
            with torch.no_grad():
                pf = prefix_forward(runner.policy, po)
                h = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                       pf.pad_masks[0].detach().cpu())
            del pf
            o = torch.cat([h, proprio(obs)])
            zn = ((o - ck["mu"]) / ck["sd"]).unsqueeze(0)
            with torch.no_grad():
                if bel is not None:
                    # matches the causal training convention: the PREVIOUS action
                    prev = (torch.zeros(1, C, ck["adim"]) if bstate is None
                            else last_u.unsqueeze(0))
                    e_ = bel.enc(zn); a_ = bel.aenc(prev.flatten(1))
                    bstate = bel.bnorm(bel.gru(
                        torch.cat([e_, a_], -1),
                        torch.zeros(1, bel.bdim) if bstate is None else bstate))
                    cin = torch.cat([zn, (bstate - ck["bmu"]) / ck["bsd"]], -1)
                else:
                    cin = zn
                chunk = pol(cin)[0]
            last_u = chunk
            for act in runner.chunk_to_env(chunk):
                obs, _r, tm, tr, inf = env.step(act); t += 1
                done = bool(tm or tr)
                if succ is None and bool(inf.get("is_success", False)):
                    succ = t
                if done: break
            if succ is None and predicate_bits(env, atoms).all():
                succ = t
        steps += t
        rows.append({"seed": seed, "success": succ is not None, "steps": t})
        print(f"{tag} s{seed}: succ={succ is not None!s:5s} t={t} ({steps} steps)",
              flush=True)

    k = sum(r["success"] for r in rows)
    (out / "summary.json").write_text(json.dumps(
        {"task": a.task, "tag": tag, "condition": ck["condition"], "utc": stamp,
         "policy": str(a.policy), "panel": [PANEL[0], PANEL[-1], len(PANEL)],
         "successes": k, "n": len(rows), "rate": k / len(rows),
         "cp95": [clopper_pearson_lower(k, len(rows)),
                  clopper_pearson_upper(k, len(rows))],
         "env_steps": steps, "episodes": rows,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"\n{tag}: {k}/{len(rows)} = {k/len(rows):.3f} {steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
