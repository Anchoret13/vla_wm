#!/usr/bin/env python
"""Action 2M.5 — train M0.3 and open the SEALED fresh test bank once.

    python scripts/train_v087_m03.py \
        --dev results/v086_bank/<run>/groups.pt \
        --test results/v087_testbank/<run>/groups.pt \
        --m02 results/v086_m02/best.pt --output results/v087_m03

The dev bank supplies B_boot-v3's 48 train / 8 validation groups.  Its `test`
split was opened during Action 2M.4 and is REPORT-ONLY here: it is excluded from
selection and from the new claim.  The fresh test bank is not read until the
validation checkpoint is fixed.
"""
from __future__ import annotations

import argparse, json, random, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from lcwm import v086_bank as B  # noqa: E402
from lcwm.v082_m0 import M0Config, M0Predictor, masked_prefix_mean, sha256_file  # noqa: E402
from lcwm.v087_residual import RESIDUAL_HIDDEN, ResidualWM, residual_loss  # noqa: E402

SEED, EPOCHS, LR, WD, CLIP = 0, 30, 3e-4, 1e-4, 1.0
CLAMP = 1e-4
NC, NB = len(B.CONTINUOUS_FIELDS), len(B.BINARY_FIELDS)
QI = B.CONTINUOUS_FIELDS.index(B.PRIMARY_TARGET)


def prepare(g: dict) -> dict:
    t = B.candidate_mean_targets(g)
    state = masked_prefix_mean(g["prefix_hidden"].float(), g["prefix_mask"])
    state = torch.cat([state, g["anchor_signature"].float()])
    C = len(g["candidate_ids"])
    ref = t["reference_index"]
    return {"anchor_id": g["anchor_id"], "split": g["split"], "ref": ref,
            "state": state.unsqueeze(0).expand(C, -1).contiguous(),
            "actions": g["actions_norm"].float(),
            "delta_physical": t["physical_mean"] - t["physical_mean"][ref],
            "delta_continuous": t["continuous_mean"] - t["continuous_mean"][ref],
            "S": t["S"], "delta": t["delta"]}


def fit_delta_norm(train) -> dict:
    P = torch.cat([a["delta_physical"] for a in train])
    C = torch.cat([a["delta_continuous"] for a in train])
    return {"phys": P.std(0).clamp(min=CLAMP), "cont": C.std(0).clamp(min=CLAMP)}


def anchor_loss(model, a, norm, *, actions=None):
    acts = a["actions"] if actions is None else actions
    pred = model(a["state"], acts, a["ref"])
    return residual_loss(pred, a, norm, QI)


def mean_val(model, anchors, norm) -> float:
    """Anchor-balanced: one loss per anchor, then the mean across anchors."""
    model.eval()
    with torch.no_grad():
        return float(torch.stack([anchor_loss(model, a, norm)[0]
                                  for a in anchors]).mean())


def score_metrics(name, anchors, pred_delta_fn, pred_phys_fn=None) -> dict:
    mae, reg, pd_all, td_all, pe = [], [], [], [], []
    for a in anchors:
        keep = [i for i in range(a["S"].shape[0]) if i != a["ref"]]
        pd = pred_delta_fn(a)
        td = a["delta"]
        mae.append(float((pd[keep] - td[keep]).abs().mean()))
        reg.append(float(a["S"].max() - a["S"][int(torch.argmax(pd))]))
        pd_all.append(pd[keep]); td_all.append(td[keep])
        if pred_phys_fn is not None:
            pe.append(float((pred_phys_fn(a) - a["delta_physical"]).abs().mean()))
    P, T = torch.cat(pd_all), torch.cat(td_all)
    corr = 0.0
    if P.std() > 1e-9 and T.std() > 1e-9:
        corr = float(((P - P.mean()) * (T - T.mean())).mean() / (P.std() * T.std()))
    out = {"model": name, "anchors": len(anchors),
           "effect_mae": sum(mae) / len(mae), "effect_corr": corr,
           "top1_regret": sum(reg) / len(reg)}
    if pe:
        out["physical_delta_err"] = sum(pe) / len(pe)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--m02", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    torch.manual_seed(SEED); random.seed(SEED)
    dev_t = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    a.output.mkdir(parents=True, exist_ok=True)

    raw = torch.load(a.dev, weights_only=False)
    groups = raw["groups"] if isinstance(raw, dict) else raw
    anchors = [prepare(g) for g in groups]
    tr = [x for x in anchors if x["split"] == "train"]
    va = [x for x in anchors if x["split"] == "val"]
    print(f"dev bank: train {len(tr)} / val {len(va)} "
          f"(the dev 'test' split is report-only and excluded)")

    ck = torch.load(a.m02, weights_only=False)
    cfg = M0Config(**ck["config"])
    base = M0Predictor(cfg, use_action=False)
    base.load_state_dict(ck["state_only"])
    base = base.to(dev_t)
    model = ResidualWM(base, z_dim=cfg.hidden_dim, action_steps=B.C_PREFIX,
                       action_dim=tr[0]["actions"].shape[-1], physical_dim=24,
                       continuous_dim=NC, hidden=RESIDUAL_HIDDEN).to(dev_t)
    print(f"b(s), z_s frozen from M0.2 state-only (epoch {ck['best_epoch']['state_only']}); "
          f"residual head {RESIDUAL_HIDDEN} units, "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable")

    norm = {k: v.to(dev_t) for k, v in fit_delta_norm(tr).items()}
    to_dev = lambda xs: [{k: (v.to(dev_t) if torch.is_tensor(v) else v)
                          for k, v in x.items()} for x in xs]
    tr, va = to_dev(tr), to_dev(va)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WD)
    best, curve = (float("inf"), None, -1), a.output / "metrics.jsonl"
    curve.write_text("")
    for ep in range(EPOCHS):
        order = list(range(len(tr))); random.Random(SEED + ep).shuffle(order)
        model.train(); model.base.eval()
        for i in order:
            opt.zero_grad(set_to_none=True)
            loss, _ = anchor_loss(model, tr[i], norm)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], CLIP)
            opt.step()
        v, t_ = mean_val(model, va, norm), mean_val(model, tr, norm)
        if v < best[0]:
            best = (v, {k: x.detach().cpu().clone()
                        for k, x in model.state_dict().items()}, ep)
        with curve.open("a") as fh:
            fh.write(json.dumps({"epoch": ep, "val": v, "train": t_}) + "\n")
        print(f"epoch={ep:02d} val={v:.6f} train={t_:.6f}", flush=True)

    model.load_state_dict(best[1]); model.eval()
    torch.save({"residual_state_dict": best[1], "best_epoch": best[2],
                "m02_checkpoint": str(a.m02), "config": ck["config"],
                "delta_norm": {k: v.cpu() for k, v in norm.items()},
                "residual_hidden": RESIDUAL_HIDDEN}, a.output / "best.pt")
    print(f"\nvalidation checkpoint FIXED at epoch {best[2]} (val {best[0]:.6f})")

    # ---- only now is the sealed test bank read ---------------------------
    traw = torch.load(a.test, weights_only=False)
    tgroups = traw["groups"] if isinstance(traw, dict) else traw
    te = to_dev([prepare(g) for g in tgroups])
    dev_src = {g["source_id"] for g in groups}
    te_src = {g["source_id"] for g in tgroups}
    if dev_src & te_src:
        raise SystemExit(f"HALT: dev/test share sources {sorted(dev_src & te_src)}")
    print(f"fresh test bank opened: {len(te)} anchors, source-disjoint from dev")

    with torch.no_grad():
        m03 = score_metrics(
            "M0.3_residual", te,
            lambda x: model(x["state"], x["actions"], x["ref"])["delta_continuous"][:, QI],
            lambda x: model(x["state"], x["actions"], x["ref"])["delta_physical"])
        zero = score_metrics("zero_delta_reference", te,
                             lambda x: torch.zeros_like(x["delta"]),
                             lambda x: torch.zeros_like(x["delta_physical"]))
        m02 = M0Predictor(cfg, use_action=True).to(dev_t)
        m02.load_state_dict(ck["action_conditioned"]); m02.eval()
        stats = {k: v.to(dev_t) for k, v in ck["stats"].items()}

        def m02_delta(x):
            o = m02(x["state"], x["actions"])
            c = o["outcome_raw"][:, NB:NB + NC] * stats["cont_std"] + stats["cont_mean"]
            return c[:, QI] - c[x["ref"], QI]

        def m02_phys(x):
            p = m02(x["state"], x["actions"])["physical_norm"][:, :24] * stats["phys_rms"]
            return p - p[x["ref"]]

        m02row = score_metrics("M0.2_absolute", te, m02_delta, m02_phys)
        m02rank = score_metrics(
            "M0.2_rank_head", te,
            lambda x: (lambda s: s - s[x["ref"]])(m02(x["state"], x["actions"])["rank_score"]))

        gap = 0.0
        for x in te:
            idx = torch.randperm(x["actions"].shape[0],
                                 generator=torch.Generator().manual_seed(SEED))
            gap += float(anchor_loss(model, x, norm, actions=x["actions"][idx])[0]
                         - anchor_loss(model, x, norm)[0])
        gap /= len(te)

    rows = [m03, zero, m02row, m02rank]
    summary = {"action": "2M.5", "utc": datetime.now(timezone.utc).isoformat(),
               "dev_groups": str(a.dev), "dev_sha256": sha256_file(a.dev),
               "test_groups": str(a.test), "test_sha256": sha256_file(a.test),
               "m02_checkpoint": str(a.m02), "best_epoch": best[2],
               "best_val": best[0], "residual_hidden": RESIDUAL_HIDDEN,
               "trainable_params": sum(p.numel() for p in model.parameters()
                                       if p.requires_grad),
               "held_out": rows, "action_shuffle_gap": gap,
               "loss": {"q": 1.0, "physical": 0.5, "aux": 0.25}}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== M0.3 fresh held-out (opened once) ===")
    keys = ("effect_mae", "effect_corr", "top1_regret", "physical_delta_err")
    print(f"{'model':24s}" + "".join(f"{k:>22s}" for k in keys))
    for r in rows:
        print(f"{r['model']:24s}" + "".join(
            (f"{r[k]:22.6f}" if k in r else f"{'-':>22s}") for k in keys))
    print(f"\naction-shuffle gap: {gap:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
