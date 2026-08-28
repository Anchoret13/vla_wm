#!/usr/bin/env python
"""Where does the selection signal die: candidates, T_theta, or p_succ?

    python scripts/probe_v127_where_signal_dies.py --wm <T_theta.pt> --head <p_succ.pt>

v126: the wm arm tied random (+6 -7, p=1.000) and the p_succ spread across the 8
candidates at a boundary had median 0.0027, so argmax was effectively random.
v125 had measured a spread of 0.0283 - but across SHUFFLED actions drawn from
other states, which are far more different than eight samples from the same
policy at the same state. The two numbers are not the same quantity.

This traces the signal through the chain at a fixed state, at zero environment
cost, and sweeps two knobs that could revive it:

  sigma  candidate diversity in action space
  depth  how far T_theta is rolled forward (composing it k times, committing to
         the candidate). c = 10 may simply be too short a horizon for a 250-step
         task: differences that matter 200 steps out need not be visible after 10.

Reported per (sigma, depth): action-space spread, predicted-latent spread, and
p_succ spread across candidates. Whichever collapses first is the bottleneck.
"""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.v080_bench import make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402
from train_v122_latent_wm import Transition  # noqa: E402
from train_v125_psucc_head import PSucc  # noqa: E402

C = 10
OUT = REPO / "results" / "v127_signal"


def pairwise(x):
    d = torch.cdist(x.flatten(1), x.flatten(1))
    iu = torch.triu_indices(len(x), len(x), offset=1)
    return float(d[iu[0], iu[1]].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[6500, 6501, 6502, 6503])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1.5, 3.0])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--at-step", type=int, default=30)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.wm, weights_only=False)
    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    mu, sd = ck["mu"], ck["sd"]
    hk = torch.load(a.head, weights_only=False)
    head = PSucc(hk["zdim"]); head.load_state_dict(hk["state_dict"]); head.eval()

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    env = make_v080_env(a.task)

    rows, steps = [], 0
    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        t = 0
        while t < a.at_step:                      # advance to a mid-episode state
            obs, _r, tm, tr, _i = env.step(
                runner.select_action(obs, env.task_description))
            t += 1
            if tm or tr: break
        steps += t
        po = runner._obs_to_policy_batch(obs, env.task_description)
        with torch.no_grad():
            pf = prefix_forward(runner.policy, po)
            z = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                   pf.pad_masks[0].detach().cpu())
            Zn = ((z - mu) / sd).unsqueeze(0).expand(a.n, -1)
            for sg in a.sigmas:
                cand = sample_chunks(runner.policy, po, a.n,
                                     seed=int(seed) * 31, prefix=pf,
                                     sigma=sg)[:, :C].detach().float().cpu()
                a_spread = pairwise(cand)
                zt = Zn
                for depth in range(1, max(a.depths) + 1):
                    zt = T(zt, cand)              # commit to the candidate
                    if depth in a.depths:
                        sc = torch.sigmoid(head(zt))
                        rows.append({"seed": seed, "sigma": sg, "depth": depth,
                                     "action_spread": a_spread,
                                     "latent_spread": pairwise(zt),
                                     "score_spread": float(sc.max() - sc.min()),
                                     "score_sd": float(sc.std())})
        del pf

    agg = {}
    for r in rows:
        k = (r["sigma"], r["depth"])
        agg.setdefault(k, []).append(r)
    print(f"{'sigma':>6} {'depth':>6} {'action':>9} {'latent':>9} "
          f"{'score sd':>9} {'score range':>12}")
    for (sg, dp) in sorted(agg):
        v = agg[(sg, dp)]
        m = lambda k: sum(x[k] for x in v) / len(v)
        print(f"{sg:6.1f} {dp:6d} {m('action_spread'):9.4f} {m('latent_spread'):9.4f} "
              f"{m('score_sd'):9.4f} {m('score_spread'):12.4f}")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "seeds": a.seeds, "n": a.n,
         "sigmas": a.sigmas, "depths": a.depths, "at_step": a.at_step,
         "env_steps": steps, "rows": rows,
         "note": "v126 wm arm tied random; p_succ spread across candidates was "
                 "0.0027. This locates where the signal collapses."}, indent=2))
    print(f"\n{steps} env steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
