#!/usr/bin/env python
"""Does the WM's ranking correlate with the candidates' TRUE outcomes?

    python scripts/probe_v133_rank_quality.py --ceiling <v132 summary.json> \
        --wm <T_theta.pt> --head <delta_w.pt>

GOAL ANCHOR (CLAUDE.md). object axis: this is the direct test of whether
D_theta(T_theta(z,u)) ranks actions correctly - the one thing the whole 5.2
pipeline has to do for selection to work.

v132 established that selection HAS leverage here: at t=0 an oracle takes
0.469 -> 0.688 (sigma=0) and 0.562 -> 0.875 (sigma=3), with candidates disagreeing
at 8/16 and 12/16 states. So the deployment nulls are about the scorer, not about
the absence of anything to score.

v132 recorded the true outcome of every candidate. Candidate sampling is
deterministic given the seed, so the exact same candidates are regenerated here
and scored - no new environment steps. Reported per sigma:

  within-state AUC   ranking quality where candidates actually disagree
  top-1 rate         how often argmax picks a succeeding candidate
  random / oracle    the bracket it has to land inside

An AUC at 0.5 means the model cannot rank, and the fix is upstream (latent
construction or T_theta), not more deployment runs.
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
OUT = REPO / "results" / "v133_rank"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceiling", type=Path, nargs="+", required=True)
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--head", type=Path, nargs="+", required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.wm, weights_only=False)
    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    mu, sd = ck["mu"], ck["sd"]
    heads = {}
    for hp in a.head:
        hk = torch.load(hp, weights_only=False)
        m = PSucc(hk["zdim"]); m.load_state_dict(hk["state_dict"]); m.eval()
        heads[hk.get("head", "p_succ")] = m
    print(f"heads: {list(heads)}")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)

    rows, steps = [], 0
    for cpath in a.ceiling:
        cd = json.loads(cpath.read_text())
        task, sigma, branch = cd["task"], cd["sigma"], cd["branch_at"]
        env = make_v080_env(task)
        pairs = {hn: {dp: [] for dp in a.depths} for hn in heads}
        for r in cd["rows"]:
            seed, outs = r["seed"], r["outcomes"]
            torch.manual_seed(seed); np.random.seed(seed)
            runner.reset(); obs, _ = env.reset(seed=int(seed))
            t = 0
            while t < branch:
                obs, _r2, tm, tr, _i = env.step(
                    runner.select_action(obs, env.task_description))
                t += 1
                if tm or tr: break
            steps += t
            po = runner._obs_to_policy_batch(obs, env.task_description)
            with torch.no_grad():
                pf = prefix_forward(runner.policy, po)
                cand = sample_chunks(runner.policy, po, len(outs),
                                     seed=int(seed) * 7919 + branch,
                                     prefix=pf, sigma=sigma)[:, :C].detach().float().cpu()
                z = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                       pf.pad_masks[0].detach().cpu())
                zt = ((z - mu) / sd).unsqueeze(0).expand(len(outs), -1)
                for dp in range(1, max(a.depths) + 1):
                    zt = T(zt, cand)
                    if dp in a.depths:
                        for hn, hm in heads.items():
                            pairs[hn][dp].append((hm(zt).numpy(), np.array(outs), seed))
            del pf

        for hn in heads:
            for dp in a.depths:
                data = pairs[hn][dp]
                mixed = [(s, o, sd_) for s, o, sd_ in data if 0 < o.sum() < len(o)]
                # within-state AUC, pooled over states where candidates disagree
                conc = disc = tie = 0
                for s, o, _ in mixed:
                    for i in range(len(o)):
                        for j in range(len(o)):
                            if o[i] > o[j]:
                                conc += s[i] > s[j]; disc += s[i] < s[j]; tie += s[i] == s[j]
                tot = conc + disc + tie
                A = (conc + 0.5 * tie) / tot if tot else float("nan")
                top1 = float(np.mean([o[int(np.argmax(s))] for s, o, _ in data]))
                rows.append({"task": task, "sigma": sigma, "head": hn, "depth": dp,
                             "within_state_auc": A, "top1": top1,
                             "n_mixed_states": len(mixed),
                             "random": cd["random_pick_rate"],
                             "oracle": cd["oracle_pick_rate"]})
                print(f"sigma={sigma} {hn:8s} depth={dp}: within-state AUC {A:.3f}  "
                      f"top-1 {top1:.3f}  (random {cd['random_pick_rate']:.3f} / "
                      f"oracle {cd['oracle_pick_rate']:.3f})", flush=True)

    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "wm": str(a.wm), "heads": [str(h) for h in a.head],
         "env_steps": steps, "rows": rows,
         "note": "candidates regenerated deterministically from the v132 seeds; "
                 "outcomes are v132's executed ground truth"}, indent=2))
    print(f"\n{steps} env steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
