#!/usr/bin/env python
"""T_theta: roll the latent forward under a candidate action. No reconstruction.

    python scripts/train_v122_latent_wm.py --tape results/v121_deploy_latents/<run>/tape.pt

GOAL ANCHOR (CLAUDE.md). This is framework 5.2's predictive object:

    z~_{t+c} = T_theta(z_t, E_a(u))

trained on trajectories collected while deploying the frozen policy, with the
target the future LATENT from PrefixVLM. No pixels are predicted or decoded.

THE DECISIVE TEST, registered before running. A model can fit z_{t+c} well while
being no world model at all, because latents are strongly autocorrelated: simply
copying z_t is already a good predictor. So the claim "T_theta predicts future
state" requires beating identity, and the claim "it is ACTION-CONDITIONED"
requires beating a no-action model that sees only z_t. Three controls:

  identity     z~ = z_t                     the autocorrelation floor
  no-action    T(z_t)                       temporal smoothing, no world model
  shuffled     T(z_t, E_a(u_permuted))      the action is real but wrong

If action-conditioned does not beat no-action on episode-disjoint held-out data,
this is a smoother and must be reported as one - not as a world model.

Predicting the RESIDUAL z_{t+c} - z_t makes the identity floor explicit rather
than hiding it inside a high R^2.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, torch  # noqa: E402
from torch import nn  # noqa: E402

OUT = REPO / "results" / "v122_latent_wm"


class ActionEncoder(nn.Module):
    def __init__(self, c: int, adim: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c * adim, dim), nn.GELU(),
                                 nn.Linear(dim, dim))

    def forward(self, u):
        return self.net(u.flatten(1))


class Transition(nn.Module):
    """T_theta. Predicts the residual in latent space; `use_action` off is the
    no-world-model control."""

    def __init__(self, zdim: int, c: int, adim: int, hidden: int = 512,
                 abed: int = 128, use_action: bool = True):
        super().__init__()
        self.use_action = use_action
        self.enc = ActionEncoder(c, adim, abed) if use_action else None
        ind = zdim + (abed if use_action else 0)
        self.net = nn.Sequential(nn.LayerNorm(ind), nn.Linear(ind, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, zdim))

    def forward(self, z, u):
        h = torch.cat([z, self.enc(u)], -1) if self.use_action else z
        return z + self.net(h)          # residual: identity is the zero solution


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--proprio-dim", type=int, default=0,
                    help="trailing dims of z that are proprioception; reported "
                         "separately because the 2048-d pooled hidden would "
                         "otherwise dominate the MSE and hide the action effect")
    a = ap.parse_args()
    d = torch.load(a.tape, weights_only=False)
    z, u, zn, ep = d["z"], d["u"], d["z_next"], d["episode"]
    print(f"{len(z)} triples over {int(ep.max())+1} episodes, latent {z.shape[-1]}, "
          f"action {tuple(u.shape[1:])}")
    torch.manual_seed(0); np.random.seed(0)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    nep = int(ep.max()) + 1
    perm = torch.randperm(nep, generator=torch.Generator().manual_seed(0))
    cut = int(nep * 0.75)
    tr_ep, te_ep = set(perm[:cut].tolist()), set(perm[cut:].tolist())
    tr = torch.tensor([i for i in range(len(z)) if int(ep[i]) in tr_ep])
    te = torch.tensor([i for i in range(len(z)) if int(ep[i]) in te_ep])
    print(f"episode-disjoint split: train {len(tr)} triples / test {len(te)}")

    mu, sd = z[tr].mean(0), z[tr].std(0) + 1e-6
    Z, ZN = (z - mu) / sd, (zn - mu) / sd

    def evaluate(model, idx, shuffle=False):
        model.eval()
        with torch.no_grad():
            uu = U[idx]
            if shuffle:
                g = torch.Generator().manual_seed(1234)
                uu = uu[torch.randperm(len(idx), generator=g)]
            pred = model(Z[idx], uu)
            mse = float(((pred - ZN[idx]) ** 2).mean())
            cos = float(nn.functional.cosine_similarity(
                pred - Z[idx], ZN[idx] - Z[idx], dim=-1).mean())
        return mse, cos

    U = u
    ident = float(((Z[te] - ZN[te]) ** 2).mean())
    delta = float(((ZN[te] - Z[te]) ** 2).mean())
    print(f"identity baseline test MSE = {ident:.5f}  (mean |delta z|^2 = {delta:.5f})")

    results = {"identity": {"mse": ident, "cos": 0.0}}
    models = {}
    for name, use_action in (("action", True), ("no_action", False)):
        m = Transition(z.shape[-1], u.shape[1], u.shape[2], a.hidden,
                       use_action=use_action)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
        best = (float("inf"), None)
        for e in range(a.epochs):
            m.train(); opt.zero_grad()
            loss = ((m(Z[tr], U[tr]) - ZN[tr]) ** 2).mean()   # latent space only
            loss.backward(); opt.step()
            if e % 10 == 0 or e == a.epochs - 1:
                mse, _ = evaluate(m, te)
                if mse < best[0]:
                    best = (mse, {k: v.clone() for k, v in m.state_dict().items()})
        m.load_state_dict(best[1]); models[name] = m
        mse, cos = evaluate(m, te)
        results[name] = {"mse": mse, "cos": cos,
                         "vs_identity": mse / ident}
        print(f"{name:10s}: test MSE {mse:.5f}  ({mse/ident:.3f} x identity)  "
              f"delta-cosine {cos:+.3f}")

    smse, scos = evaluate(models["action"], te, shuffle=True)
    results["shuffled_action"] = {"mse": smse, "cos": scos, "vs_identity": smse / ident}
    print(f"{'shuffled':10s}: test MSE {smse:.5f}  ({smse/ident:.3f} x identity)  "
          f"delta-cosine {scos:+.3f}")

    # A positive difference is not enough: test it per-triple, paired, with a
    # bootstrap over EPISODES (triples within an episode are not independent).
    def per_triple_se(model, idx, shuffle=False):
        model.eval()
        with torch.no_grad():
            uu = U[idx]
            if shuffle:
                g = torch.Generator().manual_seed(1234)
                uu = uu[torch.randperm(len(idx), generator=g)]
            return ((model(Z[idx], uu) - ZN[idx]) ** 2).mean(-1)

    se_a = per_triple_se(models["action"], te)
    se_n = per_triple_se(models["no_action"], te)
    se_s = per_triple_se(models["action"], te, shuffle=True)
    te_ep_arr = ep[te]
    uniq = sorted(set(te_ep_arr.tolist()))

    def boot(diff, iters=10000):
        g = np.random.default_rng(0)
        by = {e: diff[(te_ep_arr == e)].mean().item() for e in uniq}
        vals = np.array([by[e] for e in uniq])
        idx = g.integers(0, len(vals), size=(iters, len(vals)))
        bs = vals[idx].mean(1)
        return float(vals.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    if a.proprio_dim > 0:
        P = a.proprio_dim
        def part_mse(model, idx, sl):
            model.eval()
            with torch.no_grad():
                pred = model(Z[idx], U[idx])
                return float(((pred[:, sl] - ZN[idx][:, sl]) ** 2).mean())
        hid, pro = slice(0, Z.shape[-1] - P), slice(Z.shape[-1] - P, Z.shape[-1])
        for nm, sl in (("pooled hidden", hid), ("proprioception", pro)):
            ia = float(((Z[te][:, sl] - ZN[te][:, sl]) ** 2).mean())
            ma = part_mse(models["action"], te, sl)
            mn = part_mse(models["no_action"], te, sl)
            results[f"part_{nm.split()[0]}"] = {"identity": ia, "action": ma,
                                                "no_action": mn,
                                                "action_gain_rel": (mn - ma) / max(ia, 1e-9)}
            print(f"  [{nm:14s}] identity {ia:.5f}  action {ma:.5f} ({ma/ia:.3f}x)  "
                  f"no-action {mn:.5f} ({mn/ia:.3f}x)  action gain {100*(mn-ma)/ia:+.2f}%")

    gain, glo, ghi = boot(se_n - se_a)          # >0 means action helps
    sens, slo, shi = boot(se_s - se_a)          # >0 means the action is used
    rel = gain / ident
    results["paired"] = {"action_gain": gain, "action_gain_ci": [glo, ghi],
                         "sensitivity": sens, "sensitivity_ci": [slo, shi],
                         "n_test_episodes": len(uniq)}
    action_helps = glo > 0
    uses_action = slo > 0
    verdict = ("ACTION-CONDITIONED: action conditioning improves held-out latent "
               "prediction beyond an action-free model"
               if action_helps else
               ("USES the action but gains nothing over an action-free model - the "
                "action is redundant given z_t, which is what on-policy data gives: "
                "u is nearly a function of z. Not yet a usable world model."
                if uses_action else
                "NOT a world model: the action is ignored; this is temporal smoothing"))
    print(f"\npaired bootstrap over {len(uniq)} held-out episodes:")
    print(f"  action gain over no-action : {gain:+.5f}  95% CI [{glo:+.5f}, {ghi:+.5f}]"
          f"   ({100*rel:+.2f}% of identity)")
    print(f"  counterfactual sensitivity : {sens:+.5f}  95% CI [{slo:+.5f}, {shi:+.5f}]")
    print(verdict)

    torch.save({"state_dict": models["action"].state_dict(), "mu": mu, "sd": sd,
                "zdim": z.shape[-1], "c": u.shape[1], "adim": u.shape[2],
                "hidden": a.hidden, "task": d["task"], "results": results},
               out / "T_theta.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "tape": str(a.tape), "task": d["task"],
         "triples": len(z), "episodes": nep, "train": len(tr), "test": len(te),
         "latent_dim": int(z.shape[-1]), "epochs": a.epochs,
         "results": results, "action_gain": gain,
         "counterfactual_sensitivity": sens, "verdict": verdict, "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
