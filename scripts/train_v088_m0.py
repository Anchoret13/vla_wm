#!/usr/bin/env python
"""Action 2M.6 — bootstrap `M_0` for the closed improvement loop.

    python scripts/train_v088_m0.py --output results/v088_m0

All 72 existing groups: the eight `model_calib` groups keep their role and
select the checkpoint; the other 64 train, including the two previously opened
assessment splits whose role the action item prospectively reassigns for this
successor.  Initialized from M0.3, same reference-centered residual
factorization.

The physical term is REMOVED from the objective.  Action 2M.5 measured the
true candidate-minus-reference physical delta at exactly 0.000e+00 across all
eight fresh test anchors and bitwise-identical at 43/48 in B_boot-v3, so at
weight 0.5 it was spending a third of the objective regressing a constant.
"""
from __future__ import annotations

import argparse, json, random, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402
from lcwm import v086_bank as B  # noqa: E402
from lcwm.v082_m0 import M0Config, M0Predictor, sha256_file  # noqa: E402
from lcwm.v087_residual import RESIDUAL_HIDDEN, ResidualWM, residual_loss  # noqa: E402
from train_v087_m03 import QI, fit_delta_norm, prepare  # noqa: E402

SEED, EPOCHS, LR, WD, CLIP = 0, 60, 3e-4, 1e-4, 1.0
W_Q, W_PHYS, W_AUX = 1.0, 0.0, 0.25          # physical term removed


def loss_of(model, a, norm, *, actions=None):
    acts = a["actions"] if actions is None else actions
    pred = model(a["state"], acts, a["ref"])
    return residual_loss(pred, a, norm, QI, w_q=W_Q, w_phys=W_PHYS, w_aux=W_AUX)


def mean_loss(model, anchors, norm) -> float:
    model.eval()
    with torch.no_grad():
        return float(torch.stack([loss_of(model, a, norm)[0] for a in anchors]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path,
                    default=sorted(Path("results/v086_bank").glob("2026-*"))[-1] / "groups.pt")
    ap.add_argument("--extra", type=Path,
                    default=sorted(Path("results/v087_testbank").glob("2026-*"))[-1] / "groups.pt")
    ap.add_argument("--m03", type=Path, default=Path("results/v087_m03/best.pt"))
    ap.add_argument("--m02", type=Path, default=Path("results/v086_m02/best.pt"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    torch.manual_seed(SEED); random.seed(SEED)
    dv = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    a.output.mkdir(parents=True, exist_ok=True)

    d1 = torch.load(a.dev, weights_only=False); g1 = d1["groups"] if isinstance(d1, dict) else d1
    d2 = torch.load(a.extra, weights_only=False); g2 = d2["groups"] if isinstance(d2, dict) else d2
    allg = list(g1) + list(g2)
    anchors = [prepare(g) for g in allg]
    calib = [x for x in anchors if x["split"] == "val"]
    train = [x for x in anchors if x["split"] != "val"]
    ids = [g["source_id"] for g in allg]
    assert len(set(ids)) == len(ids), "duplicate source across the merged banks"
    assert len(anchors) == 72 and len(calib) == 8 and len(train) == 64, \
        f"expected 72/8/64, got {len(anchors)}/{len(calib)}/{len(train)}"
    print(f"merged bank: {len(anchors)} groups -> train {len(train)} / calib {len(calib)} "
          f"(calib keeps its model_calib role; the two opened assessment splits are "
          f"prospectively reassigned to training)")

    ck2 = torch.load(a.m02, weights_only=False)
    cfg = M0Config(**ck2["config"])
    base = M0Predictor(cfg, use_action=False); base.load_state_dict(ck2["state_only"])
    model = ResidualWM(base.to(dv), z_dim=cfg.hidden_dim, action_steps=B.C_PREFIX,
                       action_dim=train[0]["actions"].shape[-1], physical_dim=24,
                       continuous_dim=len(B.CONTINUOUS_FIELDS),
                       hidden=RESIDUAL_HIDDEN).to(dv)
    ck3 = torch.load(a.m03, weights_only=False)
    model.load_state_dict(ck3["residual_state_dict"])
    print(f"initialized from M0.3 (epoch {ck3['best_epoch']}); "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable")

    norm = {k: v.to(dv) for k, v in fit_delta_norm(train).items()}
    to = lambda xs: [{k: (v.to(dv) if torch.is_tensor(v) else v) for k, v in x.items()}
                     for x in xs]
    train, calib = to(train), to(calib)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WD)
    best, curve = (float("inf"), None, -1), a.output / "metrics.jsonl"
    curve.write_text("")
    for ep in range(EPOCHS):
        order = list(range(len(train))); random.Random(SEED + ep).shuffle(order)
        model.train(); model.base.eval()
        for i in order:
            opt.zero_grad(set_to_none=True)
            l, _ = loss_of(model, train[i], norm)
            if not torch.isfinite(l):
                raise FloatingPointError(f"non-finite loss at epoch {ep}")
            l.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], CLIP)
            opt.step()
        v, t = mean_loss(model, calib, norm), mean_loss(model, train, norm)
        if v < best[0]:
            best = (v, {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}, ep)
        with curve.open("a") as fh:
            fh.write(json.dumps({"epoch": ep, "calib": v, "train": t}) + "\n")
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"epoch={ep:02d} calib={v:.6f} train={t:.6f}", flush=True)

    model.load_state_dict(best[1])
    torch.save({"residual_state_dict": best[1], "best_epoch": best[2],
                "config": ck2["config"], "m02_checkpoint": str(a.m02),
                "init_from_m03": str(a.m03),
                "delta_norm": {k: v.cpu() for k, v in norm.items()},
                "residual_hidden": RESIDUAL_HIDDEN,
                "loss_weights": {"q": W_Q, "physical": W_PHYS, "aux": W_AUX}},
               a.output / "best.pt")
    summary = {"action": "2M.6", "artifact": "M_0",
               "utc": datetime.now(timezone.utc).isoformat(),
               "groups": len(anchors), "train": len(train), "calib": len(calib),
               "dev_sha256": sha256_file(a.dev), "extra_sha256": sha256_file(a.extra),
               "best_epoch": best[2], "best_calib_loss": best[0], "epochs": EPOCHS,
               "loss_weights": {"q": W_Q, "physical": W_PHYS, "aux": W_AUX},
               "note": ("offline metrics are recorded but do not gate the "
                        "interaction round")}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nM_0 selected at epoch {best[2]} (calib {best[0]:.6f}) -> {a.output}/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
