#!/usr/bin/env python
"""U2 in RAW feature space — the one configuration all measurements still permit.

    python scripts/train_v185_raw_residual.py --wm <v122 T_theta.pt> \
        --reward <v181 reward.pt> --tapes ... [--no-imagination]

Why raw space. The world model's learned 256-d encoder retains 34% of the ordering
signal (raw features 0.417 vs latent 0.140 on held-out failures, bar 0.349), and
shaping it with the only deployable label makes it worse (-0.223 at weight 5) because
success-vs-failure discrimination is orthogonal to progress. So every head read off a
compressed latent was reading a degraded representation. Here nothing is compressed:
the transition predicts raw features and the reward reads raw features, and the two
are bridged by an affine renormalisation - no decoder, so §1's no-reconstruction
constraint holds.

    z~ = T(z, E_a(u_VLA + Delta))         raw 2073-d, action-conditioned
                                          (+0.0089 gain, CI [+0.0055, +0.0124])
    objective = R7(z~; l)                 rho 0.417 raw, above the 0.349 bar

REFUTING ARM: --no-imagination optimises the same Delta against the same reward at
REAL next states, never applying T. If it matches, the transition contributes nothing
and the world model is decorative - the same arm that overturned the previous claim.

The residual is bounded and zero-initialised, per §2: a bounded perturbation of a
supported chunk, VLA still the controller. The earlier run at scale 0.15 scored 0.198
against a 0.458 base while 0.01 scored 0.490, so the bound is set small.
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
from train_v122_latent_wm import Transition  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402

OUT = REPO / "results" / "v185_raw_residual"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--reward", type=Path, required=True)
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--no-imagination", action="store_true")
    ap.add_argument("--advantage", choices=["reward", "success"], default="reward",
                    help="What weights the AWR regression. 'reward' uses R7's "
                         "potential difference, which requires R7 to pass 6.1 on "
                         "this task. On chain3 it does not (rho -0.120), so "
                         "'success' falls back to the episode outcome itself - "
                         "filtered behaviour cloning. At 1% success that is ~2 "
                         "episodes to imitate out of 192, which is the label "
                         "scarcity the paradigm runs into on its own target task.")
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    tag = "no_imag" if a.no_imagination else "imag"
    out = OUT / f"{tag}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    Z, ZN, U, SUC = [], [], [], []
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        if d["latent_dim"] != 2073 or d["task"] != a.task:
            continue
        Z.append(d["z"]); ZN.append(d["z_next"]); U.append(d["u"])
        SUC.append(d["success"][d["episode"]])
    z, zn, u = torch.cat(Z), torch.cat(ZN), torch.cat(U)
    suc = torch.cat(SUC)
    print(f"{len(z)} transitions on {a.task}; "
          f"{int(suc.sum())} from successful episodes ({float(suc.mean()):.3f})")

    ck = torch.load(a.wm, weights_only=False)
    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    for p in T.parameters():
        p.requires_grad_(False)
    wmu, wsd = ck["mu"], ck["sd"]

    rk = torch.load(a.reward, weights_only=False)
    ti = rk["tasks"].index(a.task)
    e_l = rk["E"][ti]
    rmu, rsd = rk["mu"], rk["sd"]
    heads = []
    for st in rk["states"]:
        dim = z.shape[-1] + rk["E"].shape[-1]
        h = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(),
                          nn.Linear(128, 1))
        h.load_state_dict(st); h.eval()
        for p in h.parameters():
            p.requires_grad_(False)
        heads.append(h)
    print(f"reward: {len(heads)} R7 heads, instruction '{rk['tasks'][ti]}'")

    def reward(z_wm):
        """z_wm is in the transition's normalised space; R7 reads its own."""
        raw = z_wm * wsd + wmu
        x = torch.cat([(raw - rmu) / rsd,
                       e_l.unsqueeze(0).expand(len(raw), -1)], -1)
        return torch.stack([h(x).squeeze(-1) for h in heads]).mean(0)

    Zn_ = (z - wmu) / wsd
    ZNn = (zn - wmu) / wsd
    rows = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        actor = ResidualActor(z.shape[-1], u.shape[1], u.shape[2], scale=a.scale)
        opt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-4)
        g = torch.Generator().manual_seed(700 + s)
        for e in range(a.epochs):
            i = torch.randint(0, len(z), (a.batch,), generator=g)
            actor.train(); opt.zero_grad()
            if a.no_imagination:
                # MODEL-FREE control: advantage-weighted regression. A residual on
                # the action has no offline effect without a transition - only a
                # model can say what a different action would lead to - so the
                # counterpart is not "the same objective without T" but the standard
                # model-free way to use the same data: imitate the actions that were
                # ACTUALLY executed and did better than average, weighted by their
                # observed reward. The buffer contains sigma-perturbed actions, so
                # there is real variation to imitate.
                with torch.no_grad():
                    if a.advantage == "success":
                        adv = suc[i]
                    else:
                        adv = reward(ZNn[i]) - reward(Zn_[i])
                    w = torch.softmax(adv / (adv.std() + 1e-6), 0) * len(i)
                # the residual should reproduce the advantage-weighted deviation
                # of executed actions from the batch mean
                d_ = actor(Zn_[i])
                tgt = u[i] - u[i].mean(0, keepdim=True)
                loss = (w.unsqueeze(-1).unsqueeze(-1) * (d_ - tgt) ** 2).mean()
            else:
                zc, val = Zn_[i], 0.0
                for h in range(a.horizon):
                    zc = T(zc, u[i] + actor(zc))
                    val = val + (0.95 ** h) * reward(zc)
                loss = -val.mean()
            loss.backward(); opt.step()
        actor.eval()
        with torch.no_grad():
            base = float(reward(T(Zn_, u)).mean())
            withd = float(reward(T(Zn_, u + actor(Zn_))).mean())
            dn = float(actor(Zn_).abs().mean())
        rows.append({"restart": s, "R_base": base, "R_residual": withd,
                     "imagined_gain": withd - base, "mean_abs_delta": dn})
        print(f"  restart {s}: R(base) {base:+.4f} -> R(+Delta) {withd:+.4f} "
              f"(gain {withd-base:+.4f}); mean|Delta| {dn:.4f}")
        torch.save({"state_dict": actor.state_dict(), "zdim": z.shape[-1],
                    "c": u.shape[1], "adim": u.shape[2], "scale": a.scale,
                    "arm": tag, "wm": str(a.wm), "reward": str(a.reward),
                    "task": a.task}, out / f"actor_{s}.pt")
    gains = np.array([r["imagined_gain"] for r in rows])
    print(f"\n{tag}: imagined gain {gains.mean():+.4f} +/- {gains.std():.4f}")
    print("Imagined gain is NOT evidence; deployment is.")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "arm": tag, "task": a.task, "wm": str(a.wm),
         "reward": str(a.reward), "horizon": a.horizon, "scale": a.scale,
         "transitions": len(z), "rows": rows, "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
