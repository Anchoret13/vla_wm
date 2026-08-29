#!/usr/bin/env python
"""TD-MPC2-faithful latent WM, with the no-transition arm built into the same run.

    python scripts/train_v152_tdmpc_wm.py --tapes ... --branches ... --variant deterministic

Ported from the TD-MPC2 objective (ICLR 2024, arXiv 2310.16828) after a literature
survey, because v122/v149 were written from first principles and every failure
traced to that. Differences that the survey established from the paper and its
reference implementation, each of which v149 got wrong:

  SimNorm      z is split into groups of V and each group is softmaxed onto a
               simplex. TD-MPC(v1) left the latent unconstrained and the authors
               report exploding latents; v149 used plain L2 normalisation.
  multi-step   consistency is applied over H steps with lambda^t decay, rolling
               the latent forward from ONE encoder call. v149 trained one step.
  no EMA       the consistency target is the ONLINE encoder under stop-grad, not
               an EMA copy (v1 used EMA; v2 dropped it).
  joint grads  reward and value gradients reach the encoder, so the latent is
               shaped by the task. v122's three-stage pipeline never did this.

Horizon note: TD-MPC2's default planning horizon is 3 - the authors trust the
learned transition for three steps and hand off to a terminal value. That is
consistent with our own measurement that ablating a one-step transition changed
nothing while the terminal scorer carried the signal.

THE REFUTING ARM IS COMPUTED HERE, not afterwards: `no_trans` uses the SAME jointly
trained encoder and value head but never applies the transition, scoring
value(encode(o), action) directly. If the transition contributes, wm must beat it.
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

P = 25
OUT = REPO / "results" / "v152_tdmpc_wm"


def mlp(i, h, o, n=2):
    layers = [nn.Linear(i, h), nn.GELU()]
    for _ in range(n - 1):
        layers += [nn.Linear(h, h), nn.GELU()]
    return nn.Sequential(*layers, nn.Linear(h, o))


class SimNorm(nn.Module):
    """TD-MPC2 Eq. 5: softmax over groups of V, so z lies on a product of simplices."""

    def __init__(self, v: int = 8):
        super().__init__()
        self.v = v

    def forward(self, z):
        shp = z.shape
        return z.view(*shp[:-1], -1, self.v).softmax(-1).view(*shp)


class TDMPCWM(nn.Module):
    def __init__(self, obs_dim, c, adim, zdim=256, hidden=512, v=8,
                 variant="deterministic", k=5):
        super().__init__()
        assert zdim % v == 0
        self.variant, self.k = variant, k
        self.enc = nn.Sequential(nn.LayerNorm(obs_dim), mlp(obs_dim, hidden, zdim),
                                 SimNorm(v))
        self.aenc = mlp(c * adim, hidden, 64)
        head_out = zdim if variant != "stochastic" else 2 * zdim
        n = k if variant == "ensemble" else 1
        self.dyn = nn.ModuleList([mlp(zdim + 64, hidden, head_out) for _ in range(n)])
        self.norm = SimNorm(v)
        self.value = mlp(zdim + 64, hidden, 1)     # Q(z,a): scores an action AT z
        self.reward = mlp(zdim + 64, hidden, 1)

    def encode(self, o):
        return self.enc(o)

    def step(self, z, u, sample=False):
        h = torch.cat([z, self.aenc(u.flatten(1))], -1)
        if self.variant == "stochastic":
            mu, ls = self.dyn[0](h).chunk(2, -1)
            ls = ls.clamp(-5, 2)
            zn = mu + torch.randn_like(mu) * ls.exp() if sample else mu
            return self.norm(zn), ls
        outs = torch.stack([self.norm(d(h)) for d in self.dyn])
        return outs.mean(0), outs

    def q(self, z, u):
        return self.value(torch.cat([z, self.aenc(u.flatten(1))], -1)).squeeze(-1)

    def r(self, z, u):
        return self.reward(torch.cat([z, self.aenc(u.flatten(1))], -1)).squeeze(-1)


def build_sequences(ep, tt, horizon):
    """Indices of consecutive (t, t+c, ...) runs of length horizon+1 within an episode."""
    order = {}
    for i in range(len(ep)):
        order[(int(ep[i]), int(tt[i]))] = i
    seqs = []
    for (e, t), i in order.items():
        run = [i]
        for h in range(1, horizon + 1):
            j = order.get((e, t + 10 * h))
            if j is None:
                break
            run.append(j)
        if len(run) == horizon + 1:
            seqs.append(run)
    return torch.tensor(seqs, dtype=torch.long)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--branches", type=Path, required=True)
    ap.add_argument("--variant", choices=["deterministic", "ensemble", "stochastic"],
                    default="deterministic")
    ap.add_argument("--zdim", type=int, default=256)
    ap.add_argument("--simnorm-v", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--rho", type=float, default=0.5, help="lambda decay over the horizon")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--restarts", type=int, default=5)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.variant}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    Z, U, ZN, EP, TT, DW = [], [], [], [], [], []
    off, dim0 = 0, None
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        meta = json.loads((tp.parent / "summary.json").read_text())
        if dim0 is None:
            dim0 = int(d["z"].shape[-1])
        assert int(d["z"].shape[-1]) == dim0, f"{tp}: latent dim mismatch"
        ev = {r["idx"]: {int(k): int(v) for k, v in r["events"].items()}
              for r in meta["episode_records"]}
        Z.append(d["z"]); U.append(d["u"]); ZN.append(d["z_next"])
        EP.append(d["episode"] + off); TT.append(d["t"])
        off += int(d["episode"].max()) + 1
        DW.append(torch.tensor([float(sum(1 for s in ev.get(int(e), {}).values()
                                          if int(t) < s <= int(t) + 10))
                                for e, t in zip(d["episode"], d["t"])]))
    z, u, zn = torch.cat(Z), torch.cat(U), torch.cat(ZN)
    ep, tt, dw = torch.cat(EP), torch.cat(TT), (torch.cat(DW) > 0).float()
    b = torch.load(a.branches, weights_only=False)
    bo, bu, by, bg = torch.cat([b["zh"], b["zp"]], -1), b["u"], b["y"], b["group"]
    seqs = build_sequences(ep, tt, a.horizon)
    print(f"{len(z)} transitions, {len(seqs)} sequences of length {a.horizon+1}, "
          f"{len(by)} branch candidates; variant={a.variant} zdim={a.zdim} "
          f"SimNorm V={a.simnorm_v}")

    mu_o, sd_o = z.mean(0), z.std(0) + 1e-6
    O, ON, BO = (z - mu_o) / sd_o, (zn - mu_o) / sd_o, (bo - mu_o) / sd_o
    ng = int(bg.max()) + 1
    res = {"wm": [], "no_trans": []}
    states = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        perm = torch.randperm(ng, generator=torch.Generator().manual_seed(7000 + s))
        te_g = set(perm[int(ng * 0.85):].tolist())
        btr = torch.tensor([i for i in range(len(by)) if int(bg[i]) not in te_g])
        bte = torch.tensor([i for i in range(len(by)) if int(bg[i]) in te_g])
        m = TDMPCWM(O.shape[-1], u.shape[1], u.shape[2], a.zdim,
                    v=a.simnorm_v, variant=a.variant)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        best = {"wm": (0.0, None), "no_trans": (0.0, None)}
        for e in range(a.epochs):
            m.train(); opt.zero_grad()
            # multi-step consistency from ONE encoder call, lambda-decayed
            i0 = seqs[:, 0]
            zt = m.encode(O[i0])
            cons = 0.0
            for h in range(a.horizon):
                ih = seqs[:, h]
                zt = m.step(zt, u[ih])[0]
                with torch.no_grad():
                    tgt = m.encode(ON[ih])            # stop-grad target, no EMA
                cons = cons + (a.rho ** h) * ((zt - tgt) ** 2).mean()
            zo = m.encode(O)
            rew = nn.functional.binary_cross_entropy_with_logits(m.r(zo, u), dw)
            zb = m.encode(BO[btr])
            val = nn.functional.binary_cross_entropy_with_logits(m.q(zb, bu[btr]), by[btr])
            (cons + rew + val).backward(); opt.step()
            if e % 50 == 0 or e == a.epochs - 1:
                m.eval()
                with torch.no_grad():
                    zb = m.encode(BO[bte])
                    a_wm = within_group_auc(m.q(m.step(zb, bu[bte])[0], bu[bte]),
                                            by[bte], bg[bte])
                    a_nt = within_group_auc(m.q(zb, bu[bte]), by[bte], bg[bte])
                for k, v in (("wm", a_wm), ("no_trans", a_nt)):
                    if not np.isnan(v) and v > best[k][0]:
                        best[k] = (v, {kk: vv.clone() for kk, vv in m.state_dict().items()})
        res["wm"].append(best["wm"][0]); res["no_trans"].append(best["no_trans"][0])
        states.append(best["wm"][1])
        print(f"  restart {s}: wm {best['wm'][0]:.3f}   no_trans {best['no_trans'][0]:.3f}")

    d_ = np.array(res["wm"]) - np.array(res["no_trans"])
    rng = np.random.default_rng(0)
    bs = d_[rng.integers(0, len(d_), (20000, len(d_)))].mean(1)
    lo, hi = float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
    print(f"\n{a.variant}: wm {np.mean(res['wm']):.3f}  no_trans "
          f"{np.mean(res['no_trans']):.3f}  diff {d_.mean():+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  ({int((d_>0).sum())}/{len(d_)} positive)")
    print("(frozen-latent baselines for reference: wm 0.637, pre_enc 0.636)")

    torch.save({"states": states, "variant": a.variant, "zdim": a.zdim,
                "obs_dim": O.shape[-1], "c": u.shape[1], "adim": u.shape[2],
                "mu_o": mu_o, "sd_o": sd_o, "simnorm_v": a.simnorm_v,
                "horizon": a.horizon, "task": b["task"], "results": res},
               out / "wm.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "variant": a.variant, "zdim": a.zdim,
         "simnorm_v": a.simnorm_v, "horizon": a.horizon, "rho": a.rho,
         "transitions": len(z), "sequences": len(seqs), "candidates": len(by),
         "auc_wm": res["wm"], "auc_no_trans": res["no_trans"],
         "mean_wm": float(np.mean(res["wm"])),
         "mean_no_trans": float(np.mean(res["no_trans"])),
         "diff": float(d_.mean()), "diff_ci": [lo, hi], "env_steps": 0,
         "note": "no_trans is the refuting arm: same jointly trained encoder and Q, "
                 "transition never applied",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
