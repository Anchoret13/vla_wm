#!/usr/bin/env python
"""A deploy-time world model: predict the decision-relevant outcome of a chunk.

    python scripts/train_v098_deploy_wm.py --seeds 40 --output results/v098_wm

Goal: use a WM AT DEPLOYMENT to make the VLA better, without changing policy
weights and without changing the deployment prompt.

The decisive latent on chain2b is which object the first ~40 actions commit to:
cream-first succeeds 97/104, tomato-first 0/18. So the WM only has to answer
"does this candidate chunk commit toward the cream?" - an action-conditioned
outcome prediction, which is exactly the object framework §5.2 defines.

Training labels cost ZERO environment steps. At each acquisition state we sample
one chunk under an atomic-CREAM prompt (label 1: leads to the 93%-success mode)
and one under an atomic-TOMATO prompt (label 0: leads to the 0%-success mode).
The labels come from behaviour measured across three panels, not from a guess,
and the WM never sees the atomic prompts at deployment - it scores chunks the
FULL-prompt policy proposes.

`QPhi` is deliberately NOT the target: the phase potential scores progress on the
first INCOMPLETE goal atom, and the BDDL lists tomato first, so QPhi rewards the
move that fails 0/18. Same ordered-prefix defect as §3.1.1, in the reward.
"""
from __future__ import annotations

import argparse, json, random, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from torch import nn  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.v080_bench import make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402

TASK = "chain2b_lr2"
CREAM = "pick up the cream cheese and place it in the basket"
TOMATO = "pick up the tomato sauce and place it in the basket"
C, SEED = 10, 0
ACQ = tuple(range(4900, 4990))
assert not (set(ACQ) & set(range(3200, 3400)))
OUT = REPO / "results" / "v098_wm"


class ChunkScorer(nn.Module):
    """P(chunk commits to the success mode | state, action)."""

    def __init__(self, state_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim + act_dim),
            nn.Linear(state_dim + act_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, state, chunk):
        return self.net(torch.cat([state, chunk.flatten(1)], -1)).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--per-state", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--output", type=Path, default=OUT)
    a = ap.parse_args()
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = a.output / stamp; out.mkdir(parents=True, exist_ok=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK)
    X_s, X_a, Y, G = [], [], [], []
    for si, seed in enumerate(ACQ[:a.seeds]):
        torch.manual_seed(seed); np.random.seed(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))      # reset costs no steps
        for prompt, label in ((CREAM, 1.0), (TOMATO, 0.0)):
            po = runner._obs_to_policy_batch(obs, prompt)
            with torch.no_grad():
                pf = prefix_forward(runner.policy, po)
                chunks = sample_chunks(runner.policy, po, a.per_state,
                                       seed=int(seed) * 31 + int(label),
                                       prefix=pf)[:, :C].detach().float().cpu()
                # the STATE is always encoded under the FULL prompt, because that
                # is the only prompt available at deployment
                pofull = runner._obs_to_policy_batch(obs, env.task_description)
                pff = prefix_forward(runner.policy, pofull)
                st = masked_prefix_mean(pff.hidden[0].detach().float().cpu(),
                                        pff.pad_masks[0].detach().cpu())
                del pf, pff
            for k in range(a.per_state):
                X_s.append(st); X_a.append(chunks[k]); Y.append(label); G.append(si)
        if si % 10 == 0:
            print(f"  {si+1}/{a.seeds} states, {len(Y)} labelled chunks", flush=True)

    S = torch.stack(X_s); A = torch.stack(X_a); y = torch.tensor(Y); g = torch.tensor(G)
    print(f"dataset {len(y)} chunks over {a.seeds} states, {int(y.sum())} positive")
    ns = a.seeds; cut = int(ns * 0.75)
    tr, va = g < cut, g >= cut
    print(f"state-disjoint split: train {int(tr.sum())} / val {int(va.sum())}")

    model = ChunkScorer(S.shape[-1], C * A.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = (0.0, None, -1)
    for ep in range(a.epochs):
        model.train(); opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(S[tr], A[tr]), y[tr])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = float(((model(S[va], A[va]) > 0).float() == y[va]).float().mean())
        if acc > best[0]:
            best = (acc, {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    with torch.no_grad():
        tr_acc = float(((model(S[tr], A[tr]) > 0).float() == y[tr]).float().mean())
    print(f"held-out (state-disjoint) accuracy {best[0]:.3f} at epoch {best[2]}; train {tr_acc:.3f}")

    torch.save({"state_dict": best[1], "state_dim": S.shape[-1],
                "act_dim": C * A.shape[-1], "c": C, "task": TASK,
                "val_acc": best[0]}, out / "scorer.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": TASK, "states": a.seeds, "chunks": len(y),
         "val_acc": best[0], "train_acc": tr_acc, "best_epoch": best[2],
         "env_steps": 0, "labels": {"cream": CREAM, "tomato": TOMATO},
         "note": "labels from measured behaviour: cream-first 97/104, tomato-first 0/18"},
        indent=2))
    print(f"saved -> {out}/scorer.pt   (0 environment steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
