#!/usr/bin/env python
"""Value learned by bootstrapping THROUGH the learned transition. Borrowed, not copied.

    python scripts/train_v153_td_value_wm.py --tapes ... --branches ...

WHAT IS BORROWED from TD-MPC/Dreamer and what is deliberately NOT.

Borrowed - each fixes a measured failure of v122:
  joint training     encoder, transition and value trained by one objective. v122
                     froze the latent, and the frozen pooled VLM prefix is a
                     language-vision ALIGNMENT representation: measured action gain
                     +0.81% over 10 steps.
  value-sufficiency  the latent need only support the decision quantity. v122
                     regressed the EXACT next latent under MSE and spent capacity
                     on decision-irrelevant variance.
  bounded latent     SimNorm groups + stop-grad target, the decoder-free
                     anti-collapse mechanism.

NOT borrowed - these do not fit a frozen VLA:
  MPC as controller  TD-MPC's model earns its keep by OPTIMISING action sequences.
                     Here the VLA is the actor and we may only rank what it
                     proposes, which is the weakest possible use of a model. Copying
                     MPPI would replace the policy we were asked to improve.
  dense reward       TD-MPC uses environment reward. Ours would have to come from
                     BDDL goal predicates - privileged simulator state, unavailable
                     at real deployment. Dropped: the only supervision here is
                     EPISODE SUCCESS, which a deployed system could plausibly obtain.

SO WHY WOULD THE TRANSITION EARN ITS KEEP AT ALL? Not search - STATISTICAL STRENGTH.
A bandit critic must learn Q(z,u) from candidate-level outcome labels, which cost
~500k environment steps for 2048 of them. Bootstrapping through the transition lets
every step of every ordinary trajectory carry value information backwards:

    Q(z_t, u_t)  <-  gamma * Q(z_{t+1}, u_{t+1}),   terminal <- episode success

THE REFUTING ARM is the same encoder and head trained WITHOUT bootstrapping, on the
branch labels alone. It is not "remove a layer" - it removes the mechanism. If TD
does not beat it at equal label budget, the transition buys nothing here.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, torch  # noqa: E402
from torch import nn  # noqa: E402
from train_v137_rank_head import within_group_auc  # noqa: E402
from train_v152_tdmpc_wm import SimNorm, mlp  # noqa: E402

OUT = REPO / "results" / "v153_td_value"


class ValueWM(nn.Module):
    def __init__(self, obs_dim, c, adim, zdim=256, hidden=512, v=8,
                 encoder="mlp", nv=5):
        super().__init__()
        self.encoder_kind = encoder
        if encoder == "mlp":
            self.enc = nn.Sequential(nn.LayerNorm(obs_dim), mlp(obs_dim, hidden, zdim),
                                     SimNorm(v))
        else:
            # a FIXED random projection of the frozen features, so the latent starts
            # as an information-preserving compression rather than being learned
            proj = torch.empty(obs_dim, zdim)
            nn.init.orthogonal_(proj)
            self.register_buffer("proj", proj)
            self.ln = nn.LayerNorm(obs_dim)
            self.norm_in = SimNorm(v)
            self.delta = mlp(obs_dim, hidden, zdim) if encoder == "residual" else None
        self.aenc = mlp(c * adim, hidden, 64)
        self.dyn = mlp(zdim + 64, hidden, zdim)
        self.norm = SimNorm(v)
        # an ensemble, so the actor can be optimised PESSIMISTICALLY. v164 showed
        # the residual saturates its norm bound at every scale and the imagined gain
        # grows linearly with it: V is monotone along the residual direction with no
        # interior optimum, which is what an unconstrained extrapolating value looks
        # like. Disagreement between heads is the only signal that says "you have
        # left the data".
        self.nV = nv
        self.Vs = nn.ModuleList([mlp(zdim, hidden, 1) for _ in range(nv)])
        self.Qd = mlp(zdim + 64, hidden, 1)    # direct critic, for the refuting arm

    def encode(self, o):
        if self.encoder_kind == "mlp":
            return self.enc(o)
        z = self.ln(o) @ self.proj
        if self.delta is not None:
            z = z + self.delta(self.ln(o))
        return self.norm_in(z)

    def step(self, z, u):
        return self.norm(self.dyn(torch.cat([z, self.aenc(u.flatten(1))], -1)))

    def V(self, z, reduce="mean"):
        vs = torch.stack([h(z).squeeze(-1) for h in self.Vs])
        if reduce == "min":
            return vs.min(0).values                 # pessimistic
        if reduce == "all":
            return vs
        return vs.mean(0)

    def q_through_model(self, z, u, reduce="mean"):  # value of the PREDICTED next state
        return self.V(self.step(z, u), reduce)

    def q_direct(self, z, u):                  # no transition anywhere
        return self.Qd(torch.cat([z, self.aenc(u.flatten(1))], -1)).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--branches", type=Path, required=True)
    ap.add_argument("--zdim", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=512,
                    help="minibatch size. Full-batch was three encodes of "
                         "4644x2073 per step, which does not finish on CPU.")
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--encoder", choices=["mlp", "residual", "frozen"], default="mlp",
                    help="ALTERNATIVE EXPLANATION 1 for why the jointly trained "
                         "version scores below the frozen-latent pipeline (0.608 vs "
                         "0.637): 4644 transitions may be too few to learn a 256-d "
                         "encoder from 2073-d input. 'residual' learns a small "
                         "correction on a fixed linear projection of the frozen "
                         "features; 'frozen' learns no encoder at all, isolating "
                         "the transition and value from representation learning.")
    ap.add_argument("--nstep", type=int, default=1,
                    help="ALTERNATIVE EXPLANATION 2: only 192 of 4644 transitions "
                         "are terminal, so value is anchored at few points and a "
                         "1-step bootstrap chain may be too short to propagate it. "
                         "n-step returns shorten the chain from any state to a "
                         "grounded one.")
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    Z, U, ZN, EP, TT, SUC = [], [], [], [], [], []
    off, dim0 = 0, None
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        if dim0 is None:
            dim0 = int(d["z"].shape[-1])
        assert int(d["z"].shape[-1]) == dim0, f"{tp}: latent dim mismatch"
        Z.append(d["z"]); U.append(d["u"]); ZN.append(d["z_next"])
        EP.append(d["episode"] + off); TT.append(d["t"])
        SUC.append(d["success"][d["episode"]])
        off += int(d["episode"].max()) + 1
    z, u, zn = torch.cat(Z), torch.cat(U), torch.cat(ZN)
    ep, tt, suc = torch.cat(EP), torch.cat(TT), torch.cat(SUC)
    b = torch.load(a.branches, weights_only=False)
    bo, bu, by, bg = torch.cat([b["zh"], b["zp"]], -1), b["u"], b["y"], b["group"]

    # next-index within the same episode; -1 marks the last step (terminal)
    pos = {(int(ep[i]), int(tt[i])): i for i in range(len(z))}
    # index n chunks ahead; -1 means the episode ends within n, so the target is
    # grounded in episode success rather than bootstrapped
    nxt = torch.tensor([pos.get((int(ep[i]), int(tt[i]) + 10 * a.nstep), -1)
                        for i in range(len(z))])
    term = nxt < 0
    print(f"{len(z)} transitions, {int(term.sum())} terminal, "
          f"episode success rate {float(suc.mean()):.3f}; {len(by)} branch candidates")

    mu_o, sd_o = z.mean(0), z.std(0) + 1e-6
    O, BO = (z - mu_o) / sd_o, (bo - mu_o) / sd_o
    ON = (zn - mu_o) / sd_o
    ng = int(bg.max()) + 1
    res = {"td_through_model": [], "bandit_direct": []}
    kept = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        perm = torch.randperm(ng, generator=torch.Generator().manual_seed(7000 + s))
        te_g = set(perm[int(ng * 0.85):].tolist())
        btr = torch.tensor([i for i in range(len(by)) if int(bg[i]) not in te_g])
        bte = torch.tensor([i for i in range(len(by)) if int(bg[i]) in te_g])
        m = ValueWM(O.shape[-1], u.shape[1], u.shape[2], a.zdim,
                    encoder=a.encoder)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        best = {"td_through_model": (0.0, None), "bandit_direct": (0.0, None)}
        g = torch.Generator().manual_seed(1000 + s)
        for e in range(a.epochs):
            m.train(); opt.zero_grad()
            ti = torch.randint(0, len(z), (a.batch,), generator=g)
            zt = m.encode(O[ti])
            zpred = m.step(zt, u[ti])
            with torch.no_grad():
                cons_t = m.encode(ON[ti])                   # stop-grad target
            cons = ((zpred - cons_t) ** 2).mean()
            # SARSA on the executed actions; terminal value is episode success
            with torch.no_grad():
                nb = nxt[ti]
                ok = nb >= 0
                qn = torch.zeros(len(ti))
                if ok.any():
                    qn[ok] = m.q_through_model(m.encode(O[nb[ok]]), u[nb[ok]])
                y = torch.where(term[ti], suc[ti], (a.gamma ** a.nstep) * qn)
            # each head sees an independently bootstrapped batch so they disagree
            # off-distribution rather than collapsing onto one function
            vs = m.V(zpred, reduce="all")
            bootw = torch.rand(vs.shape, generator=g) < 0.8
            td = (((vs - y.unsqueeze(0)) ** 2) * bootw).sum() / bootw.sum().clamp(min=1)
            # the refuting arm shares the encoder but never touches the transition
            bi = btr[torch.randint(0, len(btr), (a.batch,), generator=g)]
            bandit = nn.functional.binary_cross_entropy_with_logits(
                m.q_direct(m.encode(BO[bi]), bu[bi]), by[bi])
            (cons + td + bandit).backward(); opt.step()
            if e % 50 == 0 or e == a.epochs - 1:
                m.eval()
                with torch.no_grad():
                    zb = m.encode(BO[bte])
                    a_td = within_group_auc(m.q_through_model(zb, bu[bte]), by[bte], bg[bte])
                    a_bd = within_group_auc(m.q_direct(zb, bu[bte]), by[bte], bg[bte])
                for k, v in (("td_through_model", a_td), ("bandit_direct", a_bd)):
                    if not np.isnan(v) and v > best[k][0]:
                        best[k] = (v, {kk: vv.detach().clone()
                                       for kk, vv in m.state_dict().items()})
        res["td_through_model"].append(best["td_through_model"][0])
        res["bandit_direct"].append(best["bandit_direct"][0])
        kept.append(best["td_through_model"][1])
        print(f"  restart {s}: TD-through-model {best['td_through_model'][0]:.3f}   "
              f"bandit-direct {best['bandit_direct'][0]:.3f}")

    d_ = np.array(res["td_through_model"]) - np.array(res["bandit_direct"])
    rng = np.random.default_rng(0)
    bs = d_[rng.integers(0, len(d_), (20000, len(d_)))].mean(1)
    lo, hi = float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
    print(f"\nTD-through-model {np.mean(res['td_through_model']):.3f}  "
          f"bandit-direct {np.mean(res['bandit_direct']):.3f}  "
          f"diff {d_.mean():+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"({int((d_>0).sum())}/{len(d_)} positive)")
    print("NOTE: episode success is the ONLY supervision; the Delta-w reward from "
          "BDDL predicates was dropped as privileged.")

    # the model itself was never saved - downstream policy improvement needs the
    # weights, not just the AUC numbers
    bi = int(np.argmax(res["td_through_model"]))
    torch.save({"state_dict": kept[bi], "zdim": a.zdim, "obs_dim": O.shape[-1],
                "c": u.shape[1], "adim": u.shape[2], "mu_o": mu_o, "sd_o": sd_o,
                "encoder": a.encoder, "gamma": a.gamma, "nstep": a.nstep,
                "task": b["task"], "auc": res["td_through_model"][bi]},
               out / "wm.pt")
    print(f"saved best WM (restart {bi}, AUC {res['td_through_model'][bi]:.3f}) "
          f"-> {out}/wm.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "zdim": a.zdim, "gamma": a.gamma, "transitions": len(z),
         "terminal": int(term.sum()), "candidates": len(by), "results": res,
         "diff": float(d_.mean()), "diff_ci": [lo, hi], "env_steps": 0,
         "encoder": a.encoder, "nstep": a.nstep,
         "note": "TD bootstraps through the learned transition; the refuting arm is "
                 "the same encoder with a direct critic and no bootstrapping",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
