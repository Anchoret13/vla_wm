#!/usr/bin/env python
"""R1 — quasimetric value to a language anchor. Framework v2 §4, candidate R1.

    python scripts/train_v170_qrl_reward.py --tapes <tape.pt>...

WHAT THIS IS AND WHY, in the framework's terms.

§4 requires a signal that is dense, obtainable outside a simulator, and passes §6.1.
R2 (hindsight terminal goals) is dead on arrival here: 97-99% of our terminal states
are failures, so hindsight teaches "the goal is wherever my policy stops". R3
(comparing a text embedding to the VLA's h_t) is rejected: h_t is a joint (o, l)
code whose language part is a constant offset within an instruction and so
contributes nothing to ordering.

R1 instead places the goal as an ANCHOR IN THE METRIC, never required to equal any
observed state:

    z      = f(h, p)                 trainable encoder, R^256
    g_l    = G(e_l)                  trainable projection of a FROZEN instruction embedding
    V(z;l) = -d(z, g_l)              d asymmetric (MRN): d>=0, d(z,z)=0, triangle ineq.

Asymmetry is not cosmetic. A block knocked off the table is one step from the
pre-grasp state; the pre-grasp state is many steps from the fallen block. A
symmetric norm must average the two, flattening the value exactly around
irreversible events - which is what our failures are made of.

QRL, not VIP, is the skeleton: QRL's local constraint "one chunk costs at most 1"
is TRUE ON FAILED ROLLOUTS, while VIP's TD term is anchored on the video's own last
frame and is therefore mis-supervised by a 97%-failure buffer.

TERMS, and what each rules out (§4.3):
  local        relu(d(z_t, z_t+1) - 1)^2      rules out NOTHING alone - d=0 satisfies
                                              it. It sets the scale: one chunk = 1.
  spread      -phi(d(z, g)), phi=x/(1+x)      rules out total collapse of z, g and d
  mismatch     relu(m - d(z, g_l'))^2         rules out the GOAL-AGNOSTIC TIMER -
                                              the label-free form of the pathology
                                              already measured
  cross        relu(m' - d(z_i, z_j))^2       rules out spuriously low cost between
                                              states no observed path connects

local+spread run as a Lagrangian so lambda is a MONITORABLE SCALAR: a runaway lambda
means the two terms are incompatible on this data, which is itself a result.

The mismatch term is VACUOUS with one instruction, so this asserts >= 2.

No outcome labels are used here. Labels appear only in the §6.1 diagnostic, which is
a separate script.
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
from train_v152_tdmpc_wm import SimNorm, mlp  # noqa: E402

OUT = REPO / "results" / "v170_qrl_reward"
EMB_CACHE = REPO / "results" / "v170_qrl_reward" / "instruction_embeddings.pt"

INSTRUCTIONS = {
    "chain1_lr2": "put the alphabet soup in the basket",
    "chain1b_lr2": "put the tomato sauce in the basket",
    "chain2_lr2": "put the alphabet soup and the tomato sauce in the basket",
    "chain2b_lr2": "put the tomato sauce and the cream cheese in the basket",
    "chain3_lr2": "put the alphabet soup and the tomato sauce and the cream cheese "
                  "in the basket",
}


def instruction_embeddings(tasks):
    """Mean-pooled token embeddings from the VLA's OWN frozen embedding table.

    The same model family that produced h, so the instructions share a vocabulary
    space; no external encoder and no download. Cached, because loading the policy
    costs more than everything else here."""
    if EMB_CACHE.exists():
        c = torch.load(EMB_CACHE, weights_only=False)
        if all(t in c for t in tasks):
            return c
    from lcwm.libero_paths import ensure_project_libero_config
    ensure_project_libero_config()
    from transformers import AutoTokenizer
    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    # the tokenizer is not an attribute of the policy; it is named by the
    # preprocessor's TokenizerProcessorStep and resolved from the hub cache
    name = next(st.tokenizer_name for st in runner.preprocessor
                if getattr(st, "tokenizer_name", None))
    tok = AutoTokenizer.from_pretrained(name)
    emb = runner.policy.model.paligemma_with_expert.paligemma.get_input_embeddings()
    out = {}
    with torch.no_grad():
        for t in tasks:
            ids = tok(INSTRUCTIONS[t], return_tensors="pt")["input_ids"]
            out[t] = emb(ids.to(emb.weight.device))[0].float().mean(0).cpu()
    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, EMB_CACHE)
    return out


class MRN(nn.Module):
    """Metric Residual Network: an asymmetric quasimetric.

    d(x,y) = ||relu(psi(x) - psi(y))||  +  ||phi(x) - phi(y)||
             ^ asymmetric, one-directional cost   ^ symmetric part
    d >= 0 and d(x,x) = 0 by construction; both parts satisfy the triangle
    inequality, so their sum does."""

    def __init__(self, zdim, hidden=256, k=64):
        super().__init__()
        self.psi = mlp(zdim, hidden, k)
        self.phi = mlp(zdim, hidden, k)

    def forward(self, x, y):
        asym = torch.relu(self.psi(x) - self.psi(y)).pow(2).sum(-1).clamp(min=1e-8).sqrt()
        sym = (self.phi(x) - self.phi(y)).pow(2).sum(-1).clamp(min=1e-8).sqrt()
        return asym + sym


class QRLReward(nn.Module):
    def __init__(self, obs_dim, emb_dim, zdim=256, hidden=512, v=8):
        super().__init__()
        self.enc = nn.Sequential(nn.LayerNorm(obs_dim), mlp(obs_dim, hidden, zdim),
                                 SimNorm(v))
        self.goal = nn.Sequential(nn.LayerNorm(emb_dim), mlp(emb_dim, hidden, zdim),
                                  SimNorm(v))
        self.d = MRN(zdim, hidden)

    def encode(self, o):
        return self.enc(o)

    def g(self, e):
        return self.goal(e)

    def V(self, z, g):
        return -self.d(z, g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--zdim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.25, help="local-constraint budget")
    ap.add_argument("--restarts", type=int, default=3)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    Z, ZN, EP, TASK = [], [], [], []
    off, dim0 = 0, None
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        if dim0 is None:
            dim0 = int(d["z"].shape[-1])
        assert int(d["z"].shape[-1]) == dim0, f"{tp}: latent dim mismatch"
        Z.append(d["z"]); ZN.append(d["z_next"])
        EP.append(d["episode"] + off); off += int(d["episode"].max()) + 1
        TASK += [d["task"]] * len(d["z"])
    z, zn, ep = torch.cat(Z), torch.cat(ZN), torch.cat(EP)
    tasks = sorted(set(TASK))
    assert len(tasks) >= 2, (
        f"framework 4.3: the instruction-mismatch term is vacuous with one "
        f"instruction; got {tasks}")
    tix = torch.tensor([tasks.index(t) for t in TASK])
    print(f"{len(z)} transitions over {len(tasks)} instructions: {tasks}")

    embs = instruction_embeddings(tasks)
    E = torch.stack([embs[t] for t in tasks])
    E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    mu_o, sd_o = z.mean(0), z.std(0) + 1e-6
    O, ON = (z - mu_o) / sd_o, (zn - mu_o) / sd_o

    results = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        m = QRLReward(O.shape[-1], E.shape[-1], a.zdim)
        log_lam = torch.zeros(1, requires_grad=True)
        opt = torch.optim.AdamW(list(m.parameters()), lr=3e-4, weight_decay=1e-4)
        opt_l = torch.optim.Adam([log_lam], lr=1e-2)
        g = torch.Generator().manual_seed(100 + s)
        hist = []
        for e in range(a.epochs):
            i = torch.randint(0, len(z), (a.batch,), generator=g)
            zt, zt1 = m.encode(O[i]), m.encode(ON[i])
            gl = m.g(E[tix[i]])
            # (a) local: one committed chunk costs at most 1 - true on failures too
            local = torch.relu(m.d(zt, zt1) - 1.0).pow(2).mean()
            # (b) spread: push states away from goals as far as the local links allow
            spread = -(m.d(zt, gl) / (1.0 + m.d(zt, gl))).mean()
            # (c) mismatch: a state must NOT be close to another instruction's goal
            other = (tix[i] + torch.randint(1, len(tasks), (len(i),), generator=g)) % len(tasks)
            mism = torch.relu(a.margin - m.d(zt, m.g(E[other]))).pow(2).mean()
            # (d) cross-trajectory: states from different episodes are not adjacent
            j = i[torch.randperm(len(i), generator=g)]
            ok = ep[i] != ep[j]
            cross = (torch.relu(a.margin - m.d(zt[ok], m.encode(O[j[ok]]))).pow(2).mean()
                     if ok.any() else torch.zeros(()))
            lam = log_lam.exp()
            loss = spread + lam.detach() * (local - a.eps ** 2) + mism + cross
            opt.zero_grad(); loss.backward(); opt.step()
            # lambda ascends on constraint violation; a runaway lambda is a result
            opt_l.zero_grad()
            (-log_lam.exp() * (local.detach() - a.eps ** 2)).backward(); opt_l.step()
            if e % 200 == 0 or e == a.epochs - 1:
                hist.append({"epoch": e, "local": float(local), "spread": float(spread),
                             "mismatch": float(mism), "cross": float(cross),
                             "lambda": float(lam)})
        with torch.no_grad():
            zt = m.encode(O)
            d_own = m.d(zt, m.g(E[tix])).mean()
            d_oth = torch.stack([m.d(zt, m.g(E[(tix + k) % len(tasks)])).mean()
                                 for k in range(1, len(tasks))]).mean()
            collapse = float(m.d(zt[:512], zt[512:1024]).mean())
        results.append({"restart": s, "history": hist,
                        "d_own_goal": float(d_own), "d_other_goal": float(d_oth),
                        "state_state_d": collapse, "final_lambda": hist[-1]["lambda"]})
        print(f"  restart {s}: lambda {hist[-1]['lambda']:.3f}  local {hist[-1]['local']:.4f}  "
              f"d(own goal) {float(d_own):.3f}  d(other goal) {float(d_oth):.3f}  "
              f"d(state,state) {collapse:.3f}")
        torch.save({"state_dict": m.state_dict(), "zdim": a.zdim, "obs_dim": O.shape[-1],
                    "emb_dim": E.shape[-1], "mu_o": mu_o, "sd_o": sd_o,
                    "tasks": tasks, "E": E}, out / f"reward_{s}.pt")

    sep = float(np.mean([r["d_other_goal"] - r["d_own_goal"] for r in results]))
    coll = float(np.mean([r["state_state_d"] for r in results]))
    lam = float(np.mean([r["final_lambda"] for r in results]))
    print(f"\ninstruction separation d(other)-d(own) = {sep:+.3f}  "
          f"(<=0 means the mismatch term failed)")
    print(f"state-state distance {coll:.3f}  (~0 means the metric collapsed)")
    print(f"final lambda {lam:.3f}  (runaway means local and spread are incompatible)")
    print("\nNOTE: none of this says the reward is real. That is 6.1, run separately.")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "tapes": [str(t) for t in a.tapes], "tasks": tasks,
         "transitions": len(z), "results": results, "separation": sep,
         "state_state_d": coll, "final_lambda": lam, "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
