#!/usr/bin/env python
"""R4 — preference reward from episode outcomes only. Framework v2 §4, candidate R4.

    python scripts/train_v177_preference_reward.py --tapes ...

R1 is dead: with 5 instructions and 26,509 transitions its metric trained cleanly
(separation +3.2, no collapse, stable lambda) and still failed §6.1 on every
measurable task - chain1b +0.095 [-0.106,+0.297], chain2b -0.107 [-0.450,+0.235],
chain3 +0.041 [-0.107,+0.181]. Per §6.2 the candidate changes; it is not tuned.

R4 keeps the same geometry - a trainable encoder, a language anchor `g_l`, an
asymmetric MRN quasimetric, `V(z;l) = -d(z, g_l)` - and replaces the unsupervised
placement of `g_l` with the one label §9 admits: **episode success**, a single
binary judgment a deployed system could plausibly obtain from a detector or a human.

    Bradley-Terry:  P(tau_i > tau_j) = sigmoid( V(z_T^i) - V(z_T^j) )
                    label 1 when tau_i succeeded and tau_j did not

**Why this is not circular with §6.1.** Training sees only success/failure. §6.1
orders FAILED episodes by `stage_reached`, which training never sees. So the
diagnostic asks a real generalisation question: does a model taught only
"succeeded vs did not" learn to rank failures by how far they got?

The label-free terms are kept, because they are what shape the metric: the local
cost constraint (true on failed rollouts, which is why QRL and not VIP) and the
cross-trajectory hinge. The instruction-mismatch term is kept so `g_l` stays
language-conditioned rather than collapsing to one shared anchor.
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
from train_v170_qrl_reward import QRLReward, instruction_embeddings  # noqa: E402

OUT = REPO / "results" / "v177_preference_reward"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--zdim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--pairs", type=int, default=256, help="preference pairs per step")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--max-goal-d", type=float, default=30.0)
    ap.add_argument("--restarts", type=int, default=3)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    Z, ZN, EP, TASK, TERM, SUC = [], [], [], [], [], []
    off, dim0 = 0, None
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        if dim0 is None:
            dim0 = int(d["z"].shape[-1])
        assert int(d["z"].shape[-1]) == dim0
        z, ep, tt = d["z"], d["episode"], d["t"]
        Z.append(z); ZN.append(d["z_next"]); EP.append(ep + off)
        TASK += [d["task"]] * len(z)
        for e in sorted(set(ep.tolist())):
            m_ = ep == e
            last = int(torch.nonzero(m_).flatten()[torch.argmax(tt[m_])])
            TERM.append(last + sum(len(x) for x in Z[:-1]))
            SUC.append(float(d["success"][e]))
        off += int(ep.max()) + 1
    z, zn, ep = torch.cat(Z), torch.cat(ZN), torch.cat(EP)
    term = torch.tensor(TERM); suc = torch.tensor(SUC)
    tasks = sorted(set(TASK))
    assert len(tasks) >= 2
    tix = torch.tensor([tasks.index(t) for t in TASK])
    pos, neg = torch.nonzero(suc > 0).flatten(), torch.nonzero(suc == 0).flatten()
    print(f"{len(z)} transitions, {len(term)} episodes ({len(pos)} success / "
          f"{len(neg)} failure) over {len(tasks)} instructions")
    assert len(pos) >= 10 and len(neg) >= 10, "not enough of both outcomes"

    embs = instruction_embeddings(tasks)
    E = torch.stack([embs[t] for t in tasks]); E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    mu_o, sd_o = z.mean(0), z.std(0) + 1e-6
    O, ON = (z - mu_o) / sd_o, (zn - mu_o) / sd_o

    results = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        m = QRLReward(O.shape[-1], E.shape[-1], a.zdim)
        log_lam = torch.zeros(1, requires_grad=True)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        opt_l = torch.optim.Adam([log_lam], lr=1e-2)
        g = torch.Generator().manual_seed(200 + s)
        hist = []
        for e in range(a.epochs):
            i = torch.randint(0, len(z), (a.batch,), generator=g)
            zt, zt1 = m.encode(O[i]), m.encode(ON[i])
            gl = m.g(E[tix[i]])
            local = torch.relu(m.d(zt, zt1) - 1.0).pow(2).mean()
            dg = m.d(zt, gl)
            tether = torch.relu(dg - a.max_goal_d).pow(2).mean()
            other = (tix[i] + torch.randint(1, len(tasks), (len(i),), generator=g)) % len(tasks)
            mism = torch.relu(a.margin + dg - m.d(zt, m.g(E[other]))).pow(2).mean()
            j = i[torch.randperm(len(i), generator=g)]
            ok = ep[i] != ep[j]
            cross = (torch.relu(a.margin - m.d(zt[ok], m.encode(O[j[ok]]))).pow(2).mean()
                     if ok.any() else torch.zeros(()))
            # Bradley-Terry on terminal latents: the ONLY labelled term
            pi_ = pos[torch.randint(0, len(pos), (a.pairs,), generator=g)]
            ni_ = neg[torch.randint(0, len(neg), (a.pairs,), generator=g)]
            zp = m.encode(O[term[pi_]]); zn_ = m.encode(O[term[ni_]])
            vp = m.V(zp, m.g(E[tix[term[pi_]]]))
            vn = m.V(zn_, m.g(E[tix[term[ni_]]]))
            pref = -nn.functional.logsigmoid(vp - vn).mean()
            lam = log_lam.exp()
            loss = (pref + lam.detach() * (local - a.eps ** 2)
                    + mism + cross + tether)
            opt.zero_grad(); loss.backward(); opt.step()
            opt_l.zero_grad()
            (-log_lam.exp() * (local.detach() - a.eps ** 2)).backward(); opt_l.step()
            if e % 200 == 0 or e == a.epochs - 1:
                with torch.no_grad():
                    acc = float((vp > vn).float().mean())
                hist.append({"epoch": e, "pref": float(pref), "pref_acc": acc,
                             "local": float(local), "mismatch": float(mism),
                             "lambda": float(lam)})
        with torch.no_grad():
            zt = m.encode(O)
            d_own = float(m.d(zt, m.g(E[tix])).mean())
            d_oth = float(torch.stack([m.d(zt, m.g(E[(tix + k) % len(tasks)])).mean()
                                       for k in range(1, len(tasks))]).mean())
            coll = float(m.d(zt[:512], zt[512:1024]).mean())
            zp = m.encode(O[term[pos]]); zn_ = m.encode(O[term[neg]])
            acc = float((m.V(zp, m.g(E[tix[term[pos]]])).mean()
                         > m.V(zn_, m.g(E[tix[term[neg]]])).mean()))
        results.append({"restart": s, "history": hist, "sep": d_oth - d_own,
                        "state_state_d": coll, "pref_acc": hist[-1]["pref_acc"]})
        print(f"  restart {s}: preference acc {hist[-1]['pref_acc']:.3f}  "
              f"separation {d_oth - d_own:+.3f}  d(state,state) {coll:.3f}  "
              f"lambda {hist[-1]['lambda']:.3f}")
        torch.save({"state_dict": m.state_dict(), "zdim": a.zdim, "obs_dim": O.shape[-1],
                    "emb_dim": E.shape[-1], "mu_o": mu_o, "sd_o": sd_o,
                    "tasks": tasks, "E": E}, out / f"reward_{s}.pt")

    pa = float(np.mean([r["pref_acc"] for r in results]))
    print(f"\npreference accuracy {pa:.3f} (training signal, NOT evidence)")
    print("The gate is §6.1: ordering FAILED episodes by stage, which training never saw.")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "tasks": tasks, "transitions": len(z),
         "episodes": len(term), "successes": int(suc.sum()),
         "results": results, "pref_acc": pa, "env_steps": 0,
         "note": "trained on episode success only; stage_reached never seen",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
