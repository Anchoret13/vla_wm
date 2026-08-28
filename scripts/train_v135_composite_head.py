#!/usr/bin/env python
"""D_theta on [predicted proprio, current hidden], then ranked against ground truth.

    python scripts/train_v135_composite_head.py --tape <tape.pt> \
        --wm-proprio <T_theta proprio-only> --ceiling <v132 summary.json>...

GOAL ANCHOR (CLAUDE.md). object axis, and this is the architecture the measurements
force rather than one chosen for convenience:

  proprioception  T_theta is a real dynamics model here - action gain +11.78% of
                  identity, 0.235x -> 0.117x, and a shuffled action scores 1.217x,
                  WORSE than not predicting. So use its PREDICTION.
  pooled hidden   action gain +0.81%; it barely moves in c steps. So use its
                  CURRENT value rather than a prediction that adds noise but no
                  action dependence.

The head input therefore varies with the candidate action through the component
that actually responds to actions. Still latent, still no reconstruction.

Then the decisive measurement, on v132's executed ground truth: within-state
ranking AUC and top-1 across candidates that are known to disagree. v133 put the
old pipeline at AUC ~0.5 while the oracle ceiling is +0.219 (sigma=0) / +0.312
(sigma=3), so ranking - not leverage - is what has to improve.
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
from torch import nn  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.v080_bench import make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402
from train_v122_latent_wm import Transition  # noqa: E402
from train_v125_psucc_head import PSucc, auc  # noqa: E402

C, P = 10, 25
OUT = REPO / "results" / "v135_composite"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", type=Path, required=True)
    ap.add_argument("--wm-proprio", type=Path, required=True)
    ap.add_argument("--ceiling", type=Path, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4])
    a = ap.parse_args()
    d = torch.load(a.tape, weights_only=False)
    meta = json.loads((a.tape.parent / "summary.json").read_text())
    ck = torch.load(a.wm_proprio, weights_only=False)
    assert ck.get("slice") == "proprio", "expected the proprio-only T_theta"
    z, u, ep, tt = d["z"], d["u"], d["episode"], d["t"]
    torch.manual_seed(0); np.random.seed(0)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    mu_p, sd_p = ck["mu"], ck["sd"]
    zh, zp = z[:, :-P], z[:, -P:]
    mu_h, sd_h = zh.mean(0), zh.std(0) + 1e-6

    def compose(zh_raw, zp_raw, uu, depth):
        zpn = (zp_raw - mu_p) / sd_p
        for _ in range(depth):
            zpn = T(zpn, uu)
        return torch.cat([(zh_raw - mu_h) / sd_h, zpn], -1)

    # labels: Delta w over the next c steps, exact from stored achievement times
    ev = {r["idx"]: {int(k): int(v) for k, v in r["events"].items()}
          for r in meta["episode_records"]}
    dw = torch.tensor([float(sum(1 for s in ev.get(int(ep[i]), {}).values()
                                 if int(tt[i]) < s <= int(tt[i]) + C))
                       for i in range(len(z))])
    ysucc = d["success"][ep]
    nep = int(ep.max()) + 1
    perm = torch.randperm(nep, generator=torch.Generator().manual_seed(0))
    tr_ep = set(perm[:int(nep * 0.75)].tolist())
    tr = torch.tensor([i for i in range(len(z)) if int(ep[i]) in tr_ep])
    te = torch.tensor([i for i in range(len(z)) if int(ep[i]) not in tr_ep])

    heads, res = {}, {}
    for hname, y in (("delta_w", (dw > 0).float()), ("p_succ", ysucc)):
        with torch.no_grad():
            X = compose(zh, zp, u, 1)
        m = PSucc(X.shape[-1])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        pw = torch.tensor(float((y[tr] == 0).sum()) / max(float((y[tr] > 0).sum()), 1.0))
        best = (0.0, None)
        for e in range(a.epochs):
            m.train(); opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(
                m(X[tr]), y[tr], pos_weight=pw).backward()
            opt.step()
            if e % 10 == 0 or e == a.epochs - 1:
                m.eval()
                with torch.no_grad():
                    A = auc(m(X[te]).numpy(), y[te].numpy())
                if A > best[0]:
                    best = (A, {k: v.clone() for k, v in m.state_dict().items()})
        m.load_state_dict(best[1]); m.eval(); heads[hname] = m
        res[hname] = {"test_auc": best[0], "positive_rate": float(y.mean())}
        print(f"{hname:8s}: held-out AUC {best[0]:.3f} (pos rate {float(y.mean()):.3f})")

    # ---- ranking against v132's executed outcomes -------------------------------
    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    rank_rows, steps = [], 0
    for cpath in a.ceiling:
        cd = json.loads(cpath.read_text())
        env = make_v080_env(cd["task"]); sigma, branch = cd["sigma"], cd["branch_at"]
        store = {hn: {dp: [] for dp in a.depths} for hn in heads}
        for r in cd["rows"]:
            seed, outs = r["seed"], r["outcomes"]
            torch.manual_seed(seed); np.random.seed(seed)
            runner.reset(); obs, _ = env.reset(seed=int(seed))
            t = 0
            while t < branch:
                obs, _r2, tm, tr2, _i = env.step(
                    runner.select_action(obs, env.task_description))
                t += 1
                if tm or tr2: break
            steps += t
            from collect_v121_deploy_latents import proprio
            po = runner._obs_to_policy_batch(obs, env.task_description)
            with torch.no_grad():
                pf = prefix_forward(runner.policy, po)
                cand = sample_chunks(runner.policy, po, len(outs),
                                     seed=int(seed) * 7919 + branch,
                                     prefix=pf, sigma=sigma)[:, :C].detach().float().cpu()
                h = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                       pf.pad_masks[0].detach().cpu())
                pr = proprio(obs)
                H = h.unsqueeze(0).expand(len(outs), -1)
                PR = pr.unsqueeze(0).expand(len(outs), -1)
                for dp in a.depths:
                    X = compose(H, PR, cand, dp)
                    for hn, hm in heads.items():
                        store[hn][dp].append((hm(X).numpy(), np.array(outs)))
            del pf
        for hn in heads:
            for dp in a.depths:
                data = store[hn][dp]
                conc = disc = tie = 0
                for s, o in data:
                    if not (0 < o.sum() < len(o)):
                        continue
                    for i in range(len(o)):
                        for j in range(len(o)):
                            if o[i] > o[j]:
                                conc += s[i] > s[j]; disc += s[i] < s[j]; tie += s[i] == s[j]
                tot = conc + disc + tie
                A = (conc + 0.5 * tie) / tot if tot else float("nan")
                top1 = float(np.mean([o[int(np.argmax(s))] for s, o in data]))
                rank_rows.append({"sigma": sigma, "head": hn, "depth": dp,
                                  "within_state_auc": A, "top1": top1,
                                  "random": cd["random_pick_rate"],
                                  "oracle": cd["oracle_pick_rate"]})
                print(f"sigma={sigma} {hn:8s} d={dp}: within-state AUC {A:.3f}  "
                      f"top-1 {top1:.3f}  (random {cd['random_pick_rate']:.3f} / "
                      f"oracle {cd['oracle_pick_rate']:.3f})", flush=True)

    torch.save({"heads": {k: v.state_dict() for k, v in heads.items()},
                "zdim": zh.shape[-1] + P, "mu_h": mu_h, "sd_h": sd_h,
                "wm_proprio": str(a.wm_proprio), "task": d["task"]},
               out / "composite.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "tape": str(a.tape), "wm_proprio": str(a.wm_proprio),
         "heads": res, "ranking": rank_rows, "env_steps": steps,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"\n{steps} env steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
