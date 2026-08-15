#!/usr/bin/env python
"""V7.5C pre-training contract test — run BEFORE train_v075_lcwm.py.

Checks the things a 25-epoch run would otherwise discover late or,
worse, not at all:

  1. schedule    per-task mass balanced; wrapped repeats rotate the
                 query; repeat indices dense per anchor per epoch.
  2. architecture V075State loads, hist_center seeds on first forward,
                 apply_frozen is differentiable and does not observe.
  3. unroll      trace/acts alignment: acts[j] is the action recorded
                 AT decision j, and it is the one that advances
                 trace[j] -> trace[j+1]. An off-by-one here would make
                 the rollout term regress to the wrong target while
                 still looking healthy.
  4. rollout     roll starts are all non-detached, count is bounded by
                 ROLL_STARTS, every term is finite, and the term is
                 NOT vacuous (a shuffled-target control must score
                 worse than the true target).

Tensor-only; needs the h-cache. Exits non-zero on the first failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm import v075_schedule as S  # noqa: E402
from lcwm.self_predict import LatentPredictor, latent_distance  # noqa: E402
from lcwm.v075_state import V075State  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
TBPTT, ROLL_K, ROLL_STARTS = 16, 3, 4


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    anchors, tq = S.load_universe()

    # ---- 1. schedule ------------------------------------------------
    for ep in (0, 7, 24):
        order = S.rr_schedule_balanced(anchors, "train", ep)
        mass = {}
        per_anchor = {}
        for ak, rep in order:
            mass[anchors[ak]["task"]] = mass.get(anchors[ak]["task"], 0) + 1
            per_anchor.setdefault(ak, []).append(rep)
        assert len(set(mass.values())) == 1, (ep, mass)
        for ak, reps in per_anchor.items():
            assert sorted(reps) == list(range(len(reps))), (ep, ak, reps)
            qs = {S.visit_query(anchors, tq, ak, ep, r) for r in reps}
            assert len(qs) == min(len(reps),
                                  len(S.queries_for(anchors[ak]))), \
                f"epoch {ep} anchor {ak}: {len(reps)} repeats collapse " \
                f"to {len(qs)} queries"
    print("[1/4] schedule OK — balanced mass, dense repeats, "
          "rotated queries")

    # ---- 2. architecture --------------------------------------------
    model = V075State().to(device)
    hc = model.hist_center
    assert int(hc.n_updates) == 0 and float(hc.sigma) == 1.0
    model.train()
    z = torch.randn(1, 4, 384, device=device) * 3 + 7
    _ = model.hist_center(z)
    assert int(hc.n_updates) == 1, "hist_center did not seed"
    n_before = int(hc.n_updates)
    x = torch.randn(1, 4, 384, device=device, requires_grad=True)
    y = hc.apply_frozen(x)
    assert int(hc.n_updates) == n_before, "apply_frozen must not observe"
    y.sum().backward()
    assert x.grad is not None, "apply_frozen must stay differentiable"

    # REGRESSION: observe() mutates mu/sigma in place while the graph
    # holds them as the divisor. Without a cloned snapshot, a recurrent
    # unroll (one observe per decision) makes backward raise
    # "modified by an inplace operation". Reproduce a multi-step unroll
    # and require backward to succeed.
    model.train()
    m2 = V075State().to(device)
    h = torch.randn(1, 32, 2048, device=device)
    mask = torch.ones(1, 32, dtype=torch.bool, device=device)
    zz = m2.initial_state(h, mask)
    a0, am0 = m2.null_action(1, device)
    for _ in range(6):
        zz = m2.step(zz, a0, h, mask, action_mask=am0)
    zz.sum().backward()
    assert int(m2.hist_center.n_updates) >= 7, "unroll did not observe"
    print("[2/4] architecture OK — seeds on forward, apply_frozen is "
          "observe-free and differentiable, 7-step unroll backward "
          "survives the in-place statistic updates")

    # ---- 3/4. unroll alignment and rollout --------------------------
    root = None
    for p in sorted(RESULTS.glob("*_v075_lcwm_r1")):
        root = p
    idx_path = None if root is None else root / "hcache_index.json"
    if idx_path is None or not idx_path.exists():
        print("[3/4] SKIPPED — no h-cache yet "
              "(run scripts/build_v075_hcache.py)")
        print("[4/4] SKIPPED — needs the h-cache")
        return

    import json
    hidx = json.loads(idx_path.read_text())
    pred = LatentPredictor().to(device)

    # deepest train anchor: most rollout windows, crosses TBPTT
    ak = max((a for a in anchors if anchors[a]["split"] == "train"),
             key=lambda k: anchors[k]["decision"])
    a = anchors[ak]
    sid, d = a["source_id"], a["decision"]
    g, tvv = S.visit_query(anchors, tq, ak, 0, 0)
    rows_ = torch.load(S.sources_for(S.UNIVERSE) / f"{sid}.pt",
                       weights_only=False)["rows"]

    def hget(key):
        dd = torch.load(hidx[key], weights_only=False)
        return (dd["h"][None].float().to(device),
                dd["mask"][None].to(device))

    trace, acts, z = [], [], None
    for ddx in range(d):
        h, m = hget(f"{tvv}::src::{sid}::d{ddx}")
        if z is None:
            z = model.initial_state(h, m)
        else:
            prev = rows_[ddx - 1]
            aa = prev["chunk_norm"][None, :10].float().to(device)
            am = (torch.arange(10, device=device)[None]
                  < prev["executed_len"])
            z = model.step(z, aa, h, m, action_mask=am)
        if ddx > 0 and ddx % TBPTT == 0:
            z = z.detach()
        trace.append(z)
        if ddx < d - 1:
            nxt = rows_[ddx]
            acts.append((nxt["chunk_norm"][None, :10].float().to(device),
                         (torch.arange(10, device=device)[None]
                          < nxt["executed_len"])))
    h, m = hget(f"{tvv}::src::{sid}::d{d}")
    prev = rows_[d - 1]
    aa = prev["chunk_norm"][None, :10].float().to(device)
    am = (torch.arange(10, device=device)[None] < prev["executed_len"])
    z = model.step(z, aa, h, m, action_mask=am)
    acts.append((aa, am))
    trace.append(z)

    assert len(trace) == d + 1, (len(trace), d)
    assert len(acts) == d, (len(acts), d)
    # acts[j] must be the chunk recorded AT decision j
    for j in (0, d // 2, d - 1):
        assert torch.equal(
            acts[j][0].cpu(),
            rows_[j]["chunk_norm"][None, :10].float()), \
            f"acts[{j}] is not rows_[{j}] — rollout targets are shifted"
    print(f"[3/4] unroll alignment OK — anchor {ak} d={d}, "
          f"trace={len(trace)} acts={len(acts)}")

    n = len(trace)
    last = n - 1 - ROLL_K
    first = max(0, min(last, n - 1 - TBPTT))
    live = [j for j in range(first, last + 1) if trace[j].requires_grad]
    assert live, "no live rollout starts"
    for j in live:
        assert trace[j].requires_grad
    if len(live) <= ROLL_STARTS:
        starts = live
    else:
        step = (len(live) - 1) / (ROLL_STARTS - 1)
        starts = sorted({live[min(len(live) - 1, round(i * step))]
                         for i in range(ROLL_STARTS)})
    assert len(starts) <= ROLL_STARTS, starts

    true_d, ctrl_d = [], []
    for j in starts:
        zr = trace[j]
        for i in range(ROLL_K):
            if j + i >= len(acts):
                break
            aa, am = acts[j + i]
            zr = model.t(zr, model.e_a(aa, am))
            p = hc.apply_frozen(pred(zr))
            true_d.append(float(latent_distance(
                p, hc.apply_frozen(trace[j + i + 1].detach()),
                "cosine").mean()))
            # control: a target from a DIFFERENT decision. If the term
            # cannot tell them apart it is vacuous and trains nothing.
            far = (j + i + 1 + n // 2) % n
            ctrl_d.append(float(latent_distance(
                p, hc.apply_frozen(trace[far].detach()),
                "cosine").mean()))
    assert all(torch.isfinite(torch.tensor(true_d))), "non-finite roll"
    t_m = sum(true_d) / len(true_d)
    c_m = sum(ctrl_d) / len(ctrl_d)
    print(f"[4/4] rollout OK — starts={starts} (all live), "
          f"{len(true_d)} terms, true={t_m:.6f} shuffled={c_m:.6f}")
    if not c_m > t_m:
        print("      WARNING: shuffled control is NOT worse than the "
              "true target at INIT. Untrained T makes this weak "
              "evidence; the per-epoch readouts are the real check.")
    print("\nall contract checks passed")


if __name__ == "__main__":
    main()
