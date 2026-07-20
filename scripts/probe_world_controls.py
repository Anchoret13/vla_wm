#!/usr/bin/env python
"""World-probe confound controls (pre-registered 2026-07-20).

QUESTION: at a DECISION POINT (before acting), can each feature stream read the
static scene layout — object positions — across unseen layouts? The v6 pooled
cross-layout R² (siglip 0.698) is suspect: pooled variance is dominated by the
manipulated object's motion, which co-moves with the gripper, so the probe may
be tracking the ARM (proprio decodes at R²0.8), not perceiving objects.

PRE-REGISTERED DESIGN (demo-split = unseen layouts throughout):
  A  token locprobe, all frames            (replicates v6 — the suspect number)
  B  q-only ridge (9-dim proprio), all frames, same metric
       -> if B ≈ A, the v6 number carries no visual information beyond arm state
  C  token locprobe evaluated on PRE-CONTACT test frames only
       (no object >5mm from its episode-start position; arm hasn't changed the
        scene) — this isolates decision-time layout perception
  D  q-only on pre-contact frames
       -> expected ≈0 (arm near home pose, uninformative); C > D by a clear
          margin is the ONLY result that certifies visual layout reading

PASS/FAIL (pre-registered): the claim "stream X supports decision-time layout
perception across layouts" requires C_x > 0.2 pooled R² AND C_x − D > 0.2.
Anything less: the v6 cross-layout claim is retracted to "manipulated-object
tracking", and the [DEC] evidence table is amended accordingly.

Metric: pooled variance-weighted R² + mean error (cm) over eval dims. Eval dims
for A/B: train-moving (>5mm train std). For C/D: dims with cross-demo INIT
variation on train demos (>5mm std of frame-0 positions) — within pre-contact
frames the only signal is layout variation across demos.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PROBE_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets/probe_set/libero_10")
STREAMS = ["siglip", "const_ll", "real_ll"]
SEEDS = (0, 1, 2)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from probe_ladder import LocProbe, ridge_fit  # noqa: E402  (reuse, keep one impl)


def load_shards():
    shards = defaultdict(list)
    for p in PROBE_DIR.glob("task*_demo*.pt"):
        s = torch.load(p, weights_only=False)
        if s["success"]:
            shards[s["task_id"]].append(s)
    for t in shards:
        shards[t].sort(key=lambda s: s["demo"])
    return shards


def split_demos(sh):
    k = max(1, int(round(len(sh) * 0.7)))
    return sh[:k], sh[k:]


def demo_arrays(shard, stream):
    tok = torch.stack([r[stream][:64].float() for r in shard["rows"]])
    pos = torch.stack([r["obj_pos"] for r in shard["rows"]])
    q = torch.stack([r["q"] for r in shard["rows"]])
    pre = (pos - pos[0:1]).norm(dim=-1).max(dim=-1).values < 0.005  # (T,)
    return tok, pos, q, pre


def pooled_r2_err(pred, yte, dims):
    res = ((yte - pred) ** 2).reshape(len(yte), -1)[:, dims]
    tot = ((yte - yte.mean(0)) ** 2).reshape(len(yte), -1)[:, dims]
    r2 = float(1 - res.sum() / tot.sum().clamp(min=1e-8))
    obj_dims = dims.reshape(-1, 3).any(-1)
    err = float((yte - pred).norm(dim=-1)[:, obj_dims].mean()) * 100
    return r2, err


def train_locprobe(ttr, ytr, seed, epochs=300, lr=1e-2):
    torch.manual_seed(seed)
    dev = ttr.device
    mu, sd = ytr.mean(0, keepdim=True), ytr.std(0, keepdim=True).clamp(min=1e-4)
    probe = LocProbe(ttr.shape[-1], ytr.shape[1]).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.mse_loss(probe(ttr), (ytr - mu) / sd).backward()
        opt.step()
    def predict(t):
        with torch.no_grad():
            return probe(t) * sd + mu
    return predict


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    shards = load_shards()
    out: dict = {s: {k: [] for k in ("A_r2", "A_cm", "B_r2", "B_cm",
                                     "C_r2", "C_cm", "D_r2", "D_cm")}
                 for s in STREAMS}
    n_pre = []

    for tid, sh in shards.items():
        tr, te = split_demos(sh)
        for stream in STREAMS:
            TR = [demo_arrays(s, stream) for s in tr]
            TE = [demo_arrays(s, stream) for s in te]
            ttr = torch.cat([a[0] for a in TR]).to(dev)
            ytr = torch.cat([a[1] for a in TR]).to(dev)
            qtr = torch.cat([a[2] for a in TR]).to(dev)
            tte = torch.cat([a[0] for a in TE]).to(dev)
            yte = torch.cat([a[1] for a in TE]).to(dev)
            qte = torch.cat([a[2] for a in TE]).to(dev)
            pre_te = torch.cat([a[3] for a in TE]).to(dev)

            moving = (ytr.std(0).reshape(-1) > 0.005)
            init_var = torch.stack([a[1][0] for a in TR]).std(0).reshape(-1) > 0.005

            if stream == STREAMS[0]:
                n_pre.append(int(pre_te.sum()))

            # A: token probe, all frames (v6 replication)
            preds = [train_locprobe(ttr, ytr, s)(tte) for s in SEEDS]
            r2s, errs = zip(*[pooled_r2_err(p, yte, moving) for p in preds])
            out[stream]["A_r2"].append(np.mean(r2s))
            out[stream]["A_cm"].append(np.mean(errs))

            # B: q-only ridge, all frames, same dims
            w, xm, ym = ridge_fit(qtr, ytr.reshape(len(ytr), -1), 1e0)
            predq = ((qte - xm) @ w + ym).reshape(yte.shape)
            r2, err = pooled_r2_err(predq, yte, moving)
            out[stream]["B_r2"].append(r2)
            out[stream]["B_cm"].append(err)

            if int(pre_te.sum()) >= 10 and bool(init_var.any()):
                # C: token probe (trained on all train frames), eval pre-contact
                r2s, errs = zip(*[pooled_r2_err(p[pre_te], yte[pre_te], init_var)
                                  for p in preds])
                out[stream]["C_r2"].append(np.mean(r2s))
                out[stream]["C_cm"].append(np.mean(errs))
                # D: q-only, eval pre-contact
                r2, err = pooled_r2_err(predq[pre_te], yte[pre_te], init_var)
                out[stream]["D_r2"].append(r2)
                out[stream]["D_cm"].append(err)

    print(f"pre-contact test frames per task: {n_pre}")
    print(f"\n{'stream':10s} {'A all-frames':>16s} {'B q-only':>16s} "
          f"{'C pre-contact':>16s} {'D q-only-pre':>16s}")
    report = {"n_pre_contact_test_frames": n_pre, "streams": {}}
    for s in STREAMS:
        o = out[s]
        row = {k: (round(float(np.mean(v)), 3) if v else None) for k, v in o.items()}
        report["streams"][s] = {**row, "n_tasks_C": len(o["C_r2"])}
        def fmt(a, b):
            return f"{np.mean(a):6.3f}[{np.mean(b):4.1f}cm]" if a else "   n/a"
        print(f"{s:10s} {fmt(o['A_r2'], o['A_cm']):>16s} {fmt(o['B_r2'], o['B_cm']):>16s} "
              f"{fmt(o['C_r2'], o['C_cm']):>16s} {fmt(o['D_r2'], o['D_cm']):>16s}")

    outp = REPO_ROOT / "results" / "probe_world_controls.json"
    outp.write_text(json.dumps(report, indent=2))
    print("\npre-registered pass rule: C > 0.2 AND C - D > 0.2")
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
