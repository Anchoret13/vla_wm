#!/usr/bin/env python
"""World/task probe ladder over the collected probe set (framework §3, §11.1).

Rows = feature streams {siglip, const_ll, real_ll} (mean-pooled image tokens),
plus e_lang as the language-prior control where applicable.

World probes : per-task ridge regression of object positions (R^2, test demos),
               proprio q (sanity).
Task probes  : global phase (t/T) and remaining-steps regression; 10-way task-ID
               linear classification; t0-vs-t1 same-scene binary discrimination.

Splits by demo index (siblings share a split, locked contract §1):
first 70% of each task's demos -> train, rest -> test. Ridge lambda swept on a
held-out slice of train.

Output: results/probe_ladder.json + printed table.
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
LAMBDAS = [1e-2, 1e0, 1e2, 1e3, 1e4]


def load_shards():
    """task_id -> [shards], NUMERICALLY sorted by demo index (audit: glob's
    lexicographic order polluted the demo split), excluding shards whose
    collection replay failed (success=False) — documented exclusion rule."""
    shards = defaultdict(list)
    excluded = []
    for p in PROBE_DIR.glob("task*_demo*.pt"):
        s = torch.load(p, weights_only=False)
        if not s["success"]:
            excluded.append(p.name)
            continue
        shards[s["task_id"]].append(s)
    for t in shards:
        shards[t].sort(key=lambda s: s["demo"])
    return shards, sorted(excluded)


def frame_matrix(shard, stream):
    """(n_frames, 2048) mean-pooled fp32 features."""
    return torch.stack([r[stream].float().mean(dim=0) for r in shard["rows"]])


def ridge_fit(x, y, lam):
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    d = xc.shape[1]
    w = torch.linalg.solve(
        xc.T @ xc + lam * torch.eye(d, device=x.device, dtype=x.dtype), xc.T @ yc)
    return w, xm, ym


def ridge_eval(w, xm, ym, x, y, moving_mask=None):
    """Mean per-dim R^2; with moving_mask, only over dims that actually vary.

    Static objects (distractors, fixtures) have ~zero test variance, which sends
    plain R^2 to huge negatives (variance-floor artifact) — world probes must be
    scored on moving dims only.
    """
    pred = (x - xm) @ w + ym
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0).clamp(min=1e-8)
    r2 = 1 - ss_res / ss_tot
    if moving_mask is not None:
        if not bool(moving_mask.any()):
            return float("nan")
        r2 = r2[moving_mask]
    return float(r2.mean())


def fit_best(xtr, ytr, xte, yte, moving_mask=None):
    n = len(xtr)
    cut = int(n * 0.85)
    best_lam, best_v = None, -1e9
    for lam in LAMBDAS:
        w, xm, ym = ridge_fit(xtr[:cut], ytr[:cut], lam)
        v = ridge_eval(w, xm, ym, xtr[cut:], ytr[cut:], moving_mask)
        if v > best_v:
            best_v, best_lam = v, lam
    w, xm, ym = ridge_fit(xtr, ytr, best_lam)
    return ridge_eval(w, xm, ym, xte, yte, moving_mask), best_lam


def split_demos(shards_t):
    k = max(1, int(round(len(shards_t) * 0.7)))
    return shards_t[:k], shards_t[k:]


class LocProbe(torch.nn.Module):
    """Soft-argmax localization probe: per-token linear score -> softmax over the
    8x8 agentview patch grid -> expected grid coords -> per-object affine to xyz.
    Mean-pooling destroys location; this is the minimal spatially-aware linear
    readout (the standard 'is position decodable' probe)."""

    def __init__(self, d: int, n_obj: int, grid: int = 8):
        super().__init__()
        self.score = torch.nn.Linear(d, n_obj)
        self.aff = torch.nn.Parameter(0.01 * torch.randn(n_obj, 3, 3))
        self.bias = torch.nn.Parameter(torch.zeros(n_obj, 3))
        rr, cc = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        self.register_buffer(
            "coords",
            torch.stack([rr, cc], -1).reshape(-1, 2).float() / (grid - 1) - 0.5)

    def forward(self, tokens):                      # (B, 64, d) agentview
        p = self.score(tokens).softmax(dim=1)        # (B, 64, n_obj)
        e = torch.einsum("btn,tc->bnc", p, self.coords)          # (B, n_obj, 2)
        e3 = torch.cat([e, torch.ones_like(e[..., :1])], -1)     # (B, n_obj, 3)
        return torch.einsum("bnc,nck->bnk", e3, self.aff) + self.bias


def locprobe_fit_eval(ttr, ytr, tte, yte, epochs=300, lr=1e-2, seeds=(0, 1, 2)):
    """ttr: (N,64,d) agentview tokens; y: (N,n_obj,3).

    Audit fixes: moving-dims mask from TRAIN variance (no test-label leakage);
    repeated seeds, returns (mean, std) over seeds. This is a spatially-aware
    probe (softmax + affine readout), NOT a strictly linear probe.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ttr, ytr, tte, yte = (v.to(dev) for v in (ttr, ytr, tte, yte))
    mu, sd = ytr.mean(0, keepdim=True), ytr.std(0, keepdim=True).clamp(min=1e-4)
    moving = ytr.std(0).reshape(-1) > 0.005  # TRAIN-defined evaluation dims
    if not bool(moving.any()):
        return float("nan"), float("nan"), float("nan")
    scores, errs = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        probe = LocProbe(ttr.shape[-1], ytr.shape[1]).to(dev)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(probe(ttr), (ytr - mu) / sd)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = probe(tte) * sd + mu
        # POOLED R^2 over train-defined moving dims (variance-weighted): per-dim
        # averaging explodes when a train-moving dim has ~zero test variance
        # (v5 lesson: -3000s). Pooled ratio is the standard multi-output R^2.
        res = ((yte - pred) ** 2).reshape(len(yte), -1)[:, moving]
        tot = ((yte - yte.mean(0)) ** 2).reshape(len(yte), -1)[:, moving]
        scores.append(float(1 - res.sum() / tot.sum().clamp(min=1e-8)))
        # physical readout: mean 3D position error (cm) on moving OBJECTS
        obj_moving = moving.reshape(-1, 3).any(-1)
        e = (yte - pred).norm(dim=-1)[:, obj_moving]     # (N, n_moving_obj)
        errs.append(float(e.mean()) * 100.0)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(errs))


def main() -> None:
    torch.manual_seed(0)
    shards, excluded = load_shards()
    print(f"tasks: {sorted(shards)}  demos/task: "
          f"{[len(shards[t]) for t in sorted(shards)]}")
    print(f"excluded (collection replay failed): {excluded}")
    meta = {
        "excluded_shards": excluded,
        "n_frames": int(sum(len(s["rows"]) for sh in shards.values() for s in sh)),
        "split_demos": {int(t): {"train": [s["demo"] for s in split_demos(sh)[0]],
                                 "test": [s["demo"] for s in split_demos(sh)[1]]}
                        for t, sh in shards.items()},
        "locprobe_seeds": [0, 1, 2],
        "lambda_grid": LAMBDAS,
        "moving_dim_rule": "train std > 5mm",
        "note": "LocProbe is a spatially-aware probe (softmax+affine), not linear",
    }
    report: dict = {"meta": meta, "world": {}, "task": {}}

    # ---- world probes: per-task object-position regression -----------------
    for stream in STREAMS:
        r2s, q_r2s, loc_r2s, loc_errs = [], [], [], []
        loc_frame_r2s, loc_frame_stds, loc_frame_errs = [], [], []
        for tid, sh in shards.items():
            tr, te = split_demos(sh)
            if not te:
                continue
            xtr = torch.cat([frame_matrix(s, stream) for s in tr])
            xte = torch.cat([frame_matrix(s, stream) for s in te])
            ytr = torch.cat([torch.stack([r["obj_pos"].reshape(-1)
                                          for r in s["rows"]]) for s in tr])
            yte = torch.cat([torch.stack([r["obj_pos"].reshape(-1)
                                          for r in s["rows"]]) for s in te])
            moving = yte.std(0) > 0.005  # dims with >5mm test std (moving objects)
            r2, _ = fit_best(xtr, ytr, xte, yte, moving_mask=moving)
            r2s.append(r2)
            # spatially-aware localization probe (agentview tokens = first 64)
            ttr = torch.cat([torch.stack([r[stream][:64].float()
                                          for r in s["rows"]]) for s in tr])
            tte = torch.cat([torch.stack([r[stream][:64].float()
                                          for r in s["rows"]]) for s in te])
            otr = torch.cat([torch.stack([r["obj_pos"] for r in s["rows"]])
                             for s in tr])
            ote = torch.cat([torch.stack([r["obj_pos"] for r in s["rows"]])
                             for s in te])
            lr2, _, lerr = locprobe_fit_eval(ttr, otr, tte, ote)
            loc_r2s.append(lr2)
            loc_errs.append(lerr)
            # frame-level split: decodability (interpolation), demo identity leaks
            # by design — answers "is position IN the tokens at all", while the
            # demo split above answers few-shot layout generalization.
            tall = torch.cat([ttr, tte])
            oall = torch.cat([otr, ote])
            g = torch.Generator().manual_seed(tid)
            perm = torch.randperm(len(tall), generator=g)
            k = int(len(tall) * 0.8)
            m_, s_, e_ = locprobe_fit_eval(
                tall[perm[:k]], oall[perm[:k]], tall[perm[k:]], oall[perm[k:]])
            loc_frame_r2s.append(m_)
            loc_frame_stds.append(s_)
            loc_frame_errs.append(e_)
            qtr = torch.cat([torch.stack([r["q"] for r in s["rows"]]) for s in tr])
            qte = torch.cat([torch.stack([r["q"] for r in s["rows"]]) for s in te])
            qr2, _ = fit_best(xtr, qtr, xte, qte)
            q_r2s.append(qr2)
        report["world"][stream] = {
            "obj_pos_R2_mean": float(np.mean(r2s)),
            "obj_pos_R2_per_task": [round(v, 3) for v in r2s],
            "obj_pos_locprobe_R2_mean": float(np.nanmean(loc_r2s)),
            "obj_pos_locprobe_R2_per_task": [round(v, 3) for v in loc_r2s],
            "obj_pos_locprobe_frame_split_R2_mean": float(np.nanmean(loc_frame_r2s)),
            "obj_pos_locprobe_frame_split_per_task": [round(v, 3)
                                                      for v in loc_frame_r2s],
            "obj_pos_locprobe_frame_split_seed_std": [round(v, 3)
                                                      for v in loc_frame_stds],
            "obj_pos_err_cm_demo_split": [round(v, 1) for v in loc_errs],
            "obj_pos_err_cm_frame_split": [round(v, 1) for v in loc_frame_errs],
            "proprio_R2_mean": float(np.mean(q_r2s)),
        }

    # ---- task probes: global phase / remaining, task-ID classification ----
    def build_global(stream):
        xs, phase, remain, tids = [], [], [], []
        split = []
        for tid, sh in shards.items():
            tr, te = split_demos(sh)
            for s, is_tr in [(s, True) for s in tr] + [(s, False) for s in te]:
                f = frame_matrix(s, stream)
                xs.append(f)
                phase.append(torch.tensor([[r["t"] / max(r["T"], 1)]
                                           for r in s["rows"]]))
                remain.append(torch.tensor([[(r["T"] - r["t"]) / 500.0]
                                            for r in s["rows"]]))
                tids.append(torch.full((len(s["rows"]),), tid))
                split.append(torch.full((len(s["rows"]),), int(is_tr)))
        x = torch.cat(xs)
        m = torch.cat(split).bool()
        return (x, torch.cat(phase).float(), torch.cat(remain).float(),
                torch.cat(tids).long(), m)

    # true task-only prior baseline (audit #6: e_lang is post-fusion, NOT a
    # language-only prior): one-hot task identity as the only feature. taskid
    # cells are 1.0 by construction and skipped; phase/remaining show how much
    # "progress" is explained by task identity alone.
    for stream in STREAMS + ["e_lang", "taskid_prior"]:
        if stream == "taskid_prior":
            _, ph, rm, tid_all, m = build_global(STREAMS[0])
            x = torch.nn.functional.one_hot(tid_all, 10).float()
            entry = {}
            r2, _ = fit_best(x[m], ph[m], x[~m], ph[~m])
            entry["phase_R2"] = round(r2, 3)
            r2, _ = fit_best(x[m], rm[m], x[~m], rm[~m])
            entry["remaining_R2"] = round(r2, 3)
            entry["taskid_acc"] = None   # 1.0 by construction
            entry["t0_vs_t1_acc"] = None
            report["task"][stream] = entry
            continue
        if stream == "e_lang":
            xs, tids, split = [], [], []
            phase, remain = [], []
            for tid, sh in shards.items():
                tr, te = split_demos(sh)
                for s, is_tr in [(s, True) for s in tr] + [(s, False) for s in te]:
                    f = torch.stack([r["e_lang"].float() for r in s["rows"]])
                    xs.append(f)
                    phase.append(torch.tensor([[r["t"] / max(r["T"], 1)]
                                               for r in s["rows"]]))
                    remain.append(torch.tensor([[(r["T"] - r["t"]) / 500.0]
                                                for r in s["rows"]]))
                    tids.append(torch.full((len(s["rows"]),), tid))
                    split.append(torch.full((len(s["rows"]),), int(is_tr)))
            x, ph, rm = torch.cat(xs), torch.cat(phase).float(), torch.cat(remain).float()
            tid_all, m = torch.cat(tids).long(), torch.cat(split).bool()
        else:
            x, ph, rm, tid_all, m = build_global(stream)

        entry = {}
        r2, _ = fit_best(x[m], ph[m], x[~m], ph[~m])
        entry["phase_R2"] = round(r2, 3)
        r2, _ = fit_best(x[m], rm[m], x[~m], rm[~m])
        entry["remaining_R2"] = round(r2, 3)

        onehot = torch.nn.functional.one_hot(tid_all, 10).float()
        w, xm, ym = ridge_fit(x[m], onehot[m], 1e0)
        pred = ((x[~m] - xm) @ w + ym).argmax(-1)
        entry["taskid_acc"] = round(float((pred == tid_all[~m]).float().mean()), 3)

        pair = (tid_all == 0) | (tid_all == 1)
        onehot2 = torch.nn.functional.one_hot(
            (tid_all[pair] == 1).long(), 2).float()
        mp = m[pair]
        w, xm, ym = ridge_fit(x[pair][mp], onehot2[mp], 1e0)
        pred = ((x[pair][~mp] - xm) @ w + ym).argmax(-1)
        gt = (tid_all[pair][~mp] == 1).long()
        entry["t0_vs_t1_acc"] = round(float((pred == gt).float().mean()), 3)
        report["task"][stream] = entry

    out = REPO_ROOT / "results" / "probe_ladder.json"
    out.write_text(json.dumps(report, indent=2))

    print("\n=== WORLD (per-task obj-pos / proprio R2) ===")
    for s, e in report["world"].items():
        print(f"{s:10s} meanpool {e['obj_pos_R2_mean']:.3f}  "
              f"loc(demo) {e['obj_pos_locprobe_R2_mean']:.3f} "
              f"[{np.mean(e['obj_pos_err_cm_demo_split']):.1f}cm]  "
              f"loc(frame) {e['obj_pos_locprobe_frame_split_R2_mean']:.3f} "
              f"[{np.mean(e['obj_pos_err_cm_frame_split']):.1f}cm]  "
              f"proprio {e['proprio_R2_mean']:.3f}")
    print("\n=== TASK (phase / remaining / 10-way ID / t0-vs-t1) ===")
    for s, e in report["task"].items():
        tid = "  (by construction)" if e["taskid_acc"] is None else \
            f"  taskID {e['taskid_acc']:.3f}  t0v1 {e['t0_vs_t1_acc']:.3f}"
        print(f"{s:12s} phase {e['phase_R2']:.3f}  remain {e['remaining_R2']:.3f}{tid}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
