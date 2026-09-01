#!/usr/bin/env python
"""Belief with an ACTION-CONDITIONED transition, and policy improvement through it.

    python scripts/train_v205_action_belief.py --tapes ... --task chain1b_lr2

WHAT THIS FIXES, and why the previous build could not do policy improvement at all.
v190's belief was made causal by shifting the action stream, so b_t sees only
u_{t-1}. Its predictor pred1(b_t) therefore forecasts e_{t+1} WITHOUT knowing the
action that causes the transition, and a model that cannot be asked "what if I did
u instead" cannot improve a policy. Framework 5.2 requires exactly that query:

    zt_{t+c} = T_th(z_t, E_a(u^i))        roll the LATENT under a CANDIDATE action
    D_th(zt_{t+c}) = p_succ               read the head off the PREDICTED latent

So belief and transition are separated:

    b_t   = GRU(b_{t-1}, [e_t, E_a(u_{t-1})])     causal - history only, no peek
    T(b_t, E_a(u_t)) -> e^_{t+1}                  the candidate action enters HERE
    imagine: b_{t+1} = GRU(b_t, [e^_{t+1}, E_a(u_t)])   roll it forward, in latent

Then the residual on the frozen VLA is trained by backpropagating through frozen
T and D - Dreamer-style actor learning, which is the half RB-VLA does not have.

PRE-REGISTERED CONTROLS, fixed before the deployment runs (CLAUDE.md: the refuting
arm runs BEFORE the claim):
  --shuffle-action  T is trained with actions drawn from a DIFFERENT episode, so
                the action input exists and carries gradient but no true information
                about the transition. If the residual trained through THIS improves
                deployment as much, the gain is not coming from the dynamics.
  --no-action   T ignores E_a(u) entirely. Measured to be DEGENERATE, not merely
                weak: zeroing the action detaches the actor from the objective, so
                the policy gradient is identically zero and no residual exists to
                deploy. Kept as a recorded fact - a transition blind to actions
                cannot do policy improvement at all - and replaced as the control
                by --shuffle-action, which keeps the gradient path alive.
  AWR           the model-free arm on the same tapes, already at 63/96 = 0.656.
                A world model that cannot beat it has not contributed.
No claim is made from the imagined objective; only deployment counts.
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
from train_v152_tdmpc_wm import SimNorm  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402

OUT = REPO / "results" / "v205_action_belief"


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class ActionBelief(nn.Module):
    def __init__(self, zdim, c, adim, bdim=256, edim=256, hidden=512,
                 use_action=True):
        super().__init__()
        self.use_action = use_action
        self.enc = nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, hidden),
                                 nn.GELU(), nn.Linear(hidden, edim), SimNorm(8))
        self.aenc = mlp(c * adim, 128, 64)
        self.gru = nn.GRUCell(edim + 64, bdim)
        self.bnorm = SimNorm(8)
        # the candidate action enters the TRANSITION, not the belief
        self.trans = nn.Sequential(nn.Linear(bdim + 64, hidden), nn.GELU(),
                                   nn.Linear(hidden, edim), SimNorm(8))
        self.inv = mlp(2 * bdim, hidden, c * adim)
        self.head = mlp(edim, 128, 1)
        self.bdim, self.edim, self.c, self.adim = bdim, edim, c, adim

    def act(self, u):
        """(...,c,adim) -> (...,64); zeroed under the no-action control."""
        a = self.aenc(u.flatten(-2))
        return torch.zeros_like(a) if not self.use_action else a

    def step(self, b, e, u_prev):
        """One belief update. u_prev is the action taken BEFORE this observation."""
        return self.bnorm(self.gru(torch.cat([e, self.act(u_prev)], -1), b))

    def roll(self, zs, us):
        """(B,T,zdim),(B,T,c,adim) -> beliefs (B,T,bdim), encodings (B,T,edim)."""
        B, T = zs.shape[0], zs.shape[1]
        e = self.enc(zs)
        up = torch.cat([torch.zeros_like(us[:, :1]), us[:, :-1]], 1)
        b = torch.zeros(B, self.bdim, device=zs.device)
        out = []
        for t in range(T):
            b = self.step(b, e[:, t], up[:, t])
            out.append(b)
        return torch.stack(out, 1), e

    def predict(self, b, u):
        """T_th: the candidate action rolls the latent forward one committed chunk."""
        return self.trans(torch.cat([b, self.act(u)], -1))


def load_episodes(tapes, task, zdim=2073, keep_step=False):
    eps = []
    for tp in tapes:
        d = torch.load(tp, weights_only=False)
        if d["latent_dim"] != zdim or d["task"] != task:
            continue
        meta = json.loads((tp.parent / "summary.json").read_text())
        rec = {r["idx"]: r for r in meta["episode_records"]}
        z, u, ep, tt = d["z"], d["u"], d["episode"], d["t"]
        for e in sorted(set(ep.tolist())):
            r = rec.get(int(e))
            if r is None:
                continue
            idx = torch.nonzero(ep == e).flatten()[torch.argsort(tt[ep == e])]
            if len(idx) < 8:
                continue
            rec_ = {"z": z[idx], "u": u[idx], "succ": bool(r["success"])}
            if keep_step and r.get("success_step") is not None:
                # which CHUNK the success landed in, for a time-to-success target
                ts = tt[idx]
                k_ = int((ts <= r["success_step"]).sum().item()) - 1
                rec_["succ_chunk"] = max(k_, 0)
            eps.append(rec_)
    return eps


def auc(score, label):
    s = np.asarray(score, dtype=float); y = np.asarray(label, dtype=bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.empty(len(s)); o = np.argsort(s, kind="mergesort")
    sr = s[o]; i = 0
    while i < len(sr):                       # average ranks over ties
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--wm-epochs", type=int, default=4000)
    ap.add_argument("--actor-epochs", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.03)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--no-action", action="store_true",
                    help="degenerate: zero action gives an identically zero policy "
                         "gradient; trains the model, then stops before the actor")
    ap.add_argument("--shuffle-action", action="store_true",
                    help="CONTROL: T sees actions from a different episode")
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    tag = "noaction" if a.no_action else ("shufact" if a.shuffle_action else "action")
    out = OUT / f"{tag}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    eps = load_episodes(a.tapes, a.task)
    allz = torch.cat([e["z"] for e in eps])
    mu, sd = allz.mean(0), allz.std(0) + 1e-6
    L = min(min(len(e["u"]) for e in eps), 24)
    Z = torch.stack([(e["z"][:L] - mu) / sd for e in eps])
    U = torch.stack([e["u"][:L] for e in eps])
    S = torch.tensor([float(e["succ"]) for e in eps])
    print(f"{len(eps)} episodes on {a.task}, L={L}, "
          f"{int(S.sum())} successful ({float(S.mean()):.3f})")
    print(f"transition action-conditioned: {not a.no_action}"
          f"{'  (CONTROL: actions shuffled across episodes)' if a.shuffle_action else ''}")

    # ---- stage 1: belief + action-conditioned transition + success head -------
    torch.manual_seed(0)
    m = ActionBelief(Z.shape[-1], U.shape[2], U.shape[3], use_action=not a.no_action)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(205)
    bce = nn.BCEWithLogitsLoss()
    for it in range(a.wm_epochs):
        i = torch.randint(0, len(Z), (a.batch,), generator=g)
        z, u, s = Z[i], U[i], S[i]
        b, e = m.roll(z, u)
        # the belief always sees the REAL u_{t-1}; only the TRANSITION's action
        # input is scrambled under the control, which isolates action-conditioning
        ua = u[torch.randperm(len(i), generator=g)] if a.shuffle_action else u
        # 1-step: the action that CAUSES the transition conditions the prediction
        l1 = ((m.predict(b[:, :-1], ua[:, :-1]) - e[:, 1:].detach()) ** 2).mean()
        # multi-step: roll the LATENT forward, feeding predictions back in
        H = 5
        bh, eh, l5 = b[:, :-H], None, 0.0
        for h in range(H):
            eh = m.predict(bh, ua[:, h:L - H + h])
            l5 = l5 + ((eh - e[:, h + 1:L - H + h + 1].detach()) ** 2).mean()
            bh = m.step(bh.flatten(0, 1), eh.flatten(0, 1),
                        u[:, h:L - H + h].flatten(0, 1)).view(*bh.shape)
        l5 = l5 / H
        linv = ((m.inv(torch.cat([b[:, :-1], b[:, 1:]], -1))
                 - u[:, :-1].flatten(2)) ** 2).mean()
        lh = bce(m.head(e.detach()).squeeze(-1),
                 s.unsqueeze(1).expand(-1, L))
        (l1 + l5 + 0.1 * linv + lh).backward()
        opt.step(); opt.zero_grad()
        if it % 1000 == 0:
            print(f"  it {it}: pred1 {float(l1):.4f} predH {float(l5):.4f} "
                  f"inv {float(linv):.4f} head {float(lh):.4f}", flush=True)
    m.eval()

    # ---- pre-registered diagnostics, BEFORE any residual is trained ----------
    perm = torch.randperm(len(Z), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        b, e = m.roll(Z, U)
        tgt = e[:, 1:]
        diag = {
            "pred1_learned": float(((m.predict(b[:, :-1], U[:, :-1]) - tgt) ** 2).mean()),
            "pred1_identity": float(((e[:, :-1] - tgt) ** 2).mean()),
            "pred1_mean": float(((e.mean((0, 1)) - tgt) ** 2).mean()),
            "pred1_shuffled_action": float(
                ((m.predict(b[:, :-1], U[perm][:, :-1]) - tgt) ** 2).mean()),
        }
        y = S.unsqueeze(1).expand(-1, L - 1).reshape(-1).numpy()
        diag["head_auc_real"] = auc(m.head(e[:, 1:]).squeeze(-1).reshape(-1).numpy(), y)
        diag["head_auc_predicted"] = auc(
            m.head(m.predict(b[:, :-1], U[:, :-1])).squeeze(-1).reshape(-1).numpy(), y)
    print("\ndiagnostics (1-step MSE, lower better; identity = 1.000x)")
    base = diag["pred1_identity"]
    for k in ("pred1_learned", "pred1_identity", "pred1_mean", "pred1_shuffled_action"):
        print(f"  {k:24s} {diag[k]:.5f}  {diag[k]/base:6.3f} x identity")
    print(f"  head AUC on REAL latents      {diag['head_auc_real']:.3f}")
    print(f"  head AUC on PREDICTED latents {diag['head_auc_predicted']:.3f}"
          "   <- the objective the actor actually maximises")

    # ---- stage 2: residual trained by BACKPROP THROUGH the frozen model ------
    if a.no_action:
        print("\nno-action arm is DEGENERATE: zeroing E_a(u) detaches the actor "
              "from the objective, so d/d_phi V == 0 and there is no residual to "
              "deploy. Recorded as a structural fact, not a weak control.")
        (out / "summary.json").write_text(json.dumps(
            {"utc": stamp, "arm": tag, "task": a.task, "episodes": len(eps), "L": L,
             "diagnostics": diag, "rows": [], "degenerate": True, "env_steps": 0},
            indent=2))
        print(f"-> {out}")
        return 0
    for p in m.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        B_, E_ = m.roll(Z, U)
    bmu, bsd = B_.reshape(-1, B_.shape[-1]).mean(0), B_.reshape(-1, B_.shape[-1]).std(0) + 1e-6
    cdim = m.edim + m.bdim
    rows = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        actor = ResidualActor(cdim, U.shape[2], U.shape[3], scale=a.scale)
        aopt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-4)
        gg = torch.Generator().manual_seed(900 + s)
        for it in range(a.actor_epochs):
            i = torch.randint(0, len(Z), (a.batch * 8,), generator=gg)
            t = torch.randint(0, L - a.horizon, (a.batch * 8,), generator=gg)
            b, e = B_[i, t], E_[i, t]
            val = 0.0
            for h in range(a.horizon):
                u = U[i, t + h]
                d = actor(torch.cat([e, (b - bmu) / bsd], -1))
                uh = u + d
                eh = m.predict(b, uh)
                val = val + (0.95 ** h) * m.head(eh).squeeze(-1)
                b = m.step(b, eh, uh)
                e = eh
            (-val.mean()).backward(); aopt.step(); aopt.zero_grad()
        actor.eval()
        with torch.no_grad():
            b0, e0 = B_[:, :-a.horizon].reshape(-1, m.bdim), E_[:, :-a.horizon].reshape(-1, m.edim)
            u0 = U[:, :-a.horizon].reshape(-1, U.shape[2], U.shape[3])
            d0 = actor(torch.cat([e0, (b0 - bmu) / bsd], -1))
            v_base = float(m.head(m.predict(b0, u0)).mean())
            v_res = float(m.head(m.predict(b0, u0 + d0)).mean())
            dn = float(d0.abs().mean())
        rows.append({"restart": s, "V_base": v_base, "V_residual": v_res,
                     "imagined_gain": v_res - v_base, "mean_abs_delta": dn})
        print(f"  restart {s}: V(base) {v_base:+.4f} -> V(+Delta) {v_res:+.4f} "
              f"(gain {v_res-v_base:+.4f}); mean|Delta| {dn:.4f}", flush=True)
        torch.save({"state_dict": actor.state_dict(), "zdim": cdim,
                    "c": U.shape[2], "adim": U.shape[3], "scale": a.scale,
                    "model": m.state_dict(), "use_action": not a.no_action,
                    "dims": (Z.shape[-1], U.shape[2], U.shape[3]),
                    "mu": mu, "sd": sd, "bmu": bmu, "bsd": bsd,
                    "task": a.task, "arm": tag}, out / f"actor_{s}.pt")
    print("\nImagined gain is NOT evidence; deployment is.")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "arm": tag, "task": a.task, "episodes": len(eps), "L": L,
         "horizon": a.horizon, "scale": a.scale, "diagnostics": diag, "rows": rows,
         "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
