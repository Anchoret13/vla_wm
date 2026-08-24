#!/usr/bin/env python
"""Action 2M.4 — train M0.2 on B_boot-v3 and open the sealed test split once.

    python scripts/train_v086_m02.py --groups results/v086_bank/<run>/groups.pt \
        --output results/v086_m02

One fixed 30-epoch schedule inherited from M0.1; no architecture or
hyperparameter sweep.  The target is the CONTINUOUS action consequence averaged
over shared-seed repeats, not a per-rollout label.
"""
from __future__ import annotations

import argparse, json, random, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from lcwm import v086_bank as B  # noqa: E402
from lcwm.v082_m0 import M0Config, M0Predictor, masked_prefix_mean, sha256_file  # noqa: E402

SEED, EPOCHS = 0, 30
LEARNING_RATE, WEIGHT_DECAY, MAX_GRAD_NORM, HIDDEN = 3e-4, 1e-4, 1.0, 256
W_PHYSICAL, W_TASK, W_PAIRED = 1.0, 1.0, 0.25
CLAMP = 1e-4
NC, NB = len(B.CONTINUOUS_FIELDS), len(B.BINARY_FIELDS)
QI = B.CONTINUOUS_FIELDS.index(B.PRIMARY_TARGET)


def prepare(g: dict) -> dict:
    """Per-anchor tensors: state, actions, and candidate-mean targets."""
    t = B.candidate_mean_targets(g)
    state = masked_prefix_mean(g["prefix_hidden"].float(), g["prefix_mask"])
    state = torch.cat([state, g["anchor_signature"].float()])
    C = len(g["candidate_ids"])
    return {"anchor_id": g["anchor_id"], "split": g["split"],
            "state": state.unsqueeze(0).expand(C, -1).contiguous(),
            "actions": g["actions_norm"].float(),
            "physical": t["physical_mean"], "continuous": t["continuous_mean"],
            "binary": t["binary_rate"], "S": t["S"], "delta": t["delta"],
            "ref": t["reference_index"]}


def fit_stats(train: list[dict]) -> dict:
    """RMS for physical deltas, mean/std for continuous targets, clamped.
    Fitted on TRAIN sources only and frozen before validation is touched."""
    P = torch.cat([a["physical"] for a in train])
    Cc = torch.cat([a["continuous"] for a in train])
    D = torch.cat([a["delta"] for a in train])
    return {"phys_rms": P.pow(2).mean(0).sqrt().clamp(min=CLAMP),
            "cont_mean": Cc.mean(0), "cont_std": Cc.std(0).clamp(min=CLAMP),
            "delta_std": D.std().clamp(min=CLAMP)}


def anchor_loss(model, a: dict, st: dict, *, actions=None) -> tuple:
    acts = a["actions"] if actions is None else actions
    out = model(a["state"], acts)
    l_phys = F.smooth_l1_loss(out["physical_norm"][:, :24],
                              a["physical"] / st["phys_rms"])
    raw = out["outcome_raw"]
    l_cont = F.smooth_l1_loss(raw[:, NB:NB + NC],
                              (a["continuous"] - st["cont_mean"]) / st["cont_std"])
    l_bin = F.binary_cross_entropy_with_logits(raw[:, :NB], a["binary"].clamp(0, 1))
    l_task = l_cont + l_bin
    score = out["rank_score"]
    pred_delta = score - score[a["ref"]]
    keep = [i for i in range(len(score)) if i != a["ref"]]
    l_pair = F.smooth_l1_loss(pred_delta[keep] / st["delta_std"],
                              a["delta"][keep] / st["delta_std"])
    total = W_PHYSICAL * l_phys + W_TASK * l_task + W_PAIRED * l_pair
    return total, {"physical": l_phys, "task": l_task, "paired": l_pair,
                   "total": total}, out


def mean_loss(model, anchors, st) -> float:
    """Averaged within an anchor, then across anchors."""
    model.eval()
    with torch.no_grad():
        return float(torch.stack([anchor_loss(model, a, st)[0]
                                  for a in anchors]).mean())


def evaluate(model, anchors, st, name: str) -> dict:
    model.eval()
    pe, ce_abs, ce_sq, dl_err, regret = [], [], [], [], []
    pred_S, true_S, dmg_e, succ_e = [], [], [], []
    with torch.no_grad():
        for a in anchors:
            _t, _p, out = anchor_loss(model, a, st)
            phys = out["physical_norm"][:, :24] * st["phys_rms"]
            pe.append(float((phys - a["physical"]).pow(2).mean()))
            cont = out["outcome_raw"][:, NB:NB + NC] * st["cont_std"] + st["cont_mean"]
            ce_abs.append(float((cont[:, QI] - a["continuous"][:, QI]).abs().mean()))
            ce_sq.append(float((cont[:, QI] - a["continuous"][:, QI]).pow(2).mean()))
            sc = out["rank_score"]
            pd = sc - sc[a["ref"]]
            dl_err.append(float((pd - a["delta"]).abs().mean()))
            pick = int(torch.argmax(pd))
            regret.append(float(a["S"].max() - a["S"][pick]))
            pred_S.append(pd); true_S.append(a["delta"])
            pr = torch.sigmoid(out["outcome_raw"][:, :NB])
            dmg_e.append(float((pr[:, 0] - a["binary"][:, 0]).abs().mean()))
            succ_e.append(float((pr[:, 1] - a["binary"][:, 1]).abs().mean()))
    P, T = torch.cat(pred_S), torch.cat(true_S)
    corr = 0.0
    if P.std() > 1e-9 and T.std() > 1e-9:
        corr = float(((P - P.mean()) * (T - T.mean())).mean() / (P.std() * T.std()))
    return {"model": name, "anchors": len(anchors),
            "physical_mse": sum(pe) / len(pe),
            "effect_mae": sum(ce_abs) / len(ce_abs),
            "effect_rmse": (sum(ce_sq) / len(ce_sq)) ** 0.5,
            "paired_effect_abs_err": sum(dl_err) / len(dl_err),
            "top1_regret": sum(regret) / len(regret),
            "effect_corr": corr,
            "dmg_cal_err": sum(dmg_e) / len(dmg_e),
            "succ_cal_err": sum(succ_e) / len(succ_e)}


def baselines(train, test) -> list[dict]:
    """Predictors that need no model: copy/no-change, train empirical mean, and
    the stock reference candidate."""
    out = []
    zero_regret = [float(a["S"].max() - a["S"][a["ref"]]) for a in test]
    out.append({"model": "stock_reference", "anchors": len(test),
                "top1_regret": sum(zero_regret) / len(zero_regret),
                "paired_effect_abs_err": sum(
                    float(a["delta"].abs().mean()) for a in test) / len(test),
                "note": "always executes the reference; delta prediction is 0"})
    mu = torch.cat([a["delta"] for a in train]).mean()
    err = [float((mu - a["delta"]).abs().mean()) for a in test]
    reg = []
    for a in test:                       # constant delta -> argmax is index 0
        reg.append(float(a["S"].max() - a["S"][0]))
    out.append({"model": "train_empirical_mean", "anchors": len(test),
                "paired_effect_abs_err": sum(err) / len(err),
                "top1_regret": sum(reg) / len(reg),
                "note": f"constant predicted delta = {float(mu):+.5f}"})
    physc = [float(a["physical"].pow(2).mean()) for a in test]
    contc = [float((a["continuous"][:, QI] - a["continuous"][a["ref"], QI]).abs().mean())
             for a in test]
    out.append({"model": "copy_no_change", "anchors": len(test),
                "physical_mse": sum(physc) / len(physc),
                "effect_mae": sum(contc) / len(contc),
                "note": "predicts zero physical delta and the reference's consequence"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    torch.manual_seed(SEED); random.seed(SEED)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    a.output.mkdir(parents=True, exist_ok=True)

    raw = torch.load(a.groups, weights_only=False)
    groups = raw["groups"] if isinstance(raw, dict) else raw
    anchors = [prepare(g) for g in groups]
    sp = {k: [x for x in anchors if x["split"] == k] for k in ("train", "val", "test")}
    src = {k: {g["source_id"] for g in groups if g["split"] == k}
           for k in ("train", "val", "test")}
    for x in ("train", "val"), ("train", "test"), ("val", "test"):
        if src[x[0]] & src[x[1]]:
            raise SystemExit(f"HALT: {x[0]}/{x[1]} share sources")
    print({k: len(v) for k, v in sp.items()}, "| source-disjoint: OK")

    st = fit_stats(sp["train"])                 # TRAIN ONLY, frozen here
    st = {k: v.to(dev) for k, v in st.items()}
    for k in sp:
        sp[k] = [{kk: (vv.to(dev) if torch.is_tensor(vv) else vv)
                  for kk, vv in x.items()} for x in sp[k]]

    cfg = M0Config(state_dim=sp["train"][0]["state"].shape[-1],
                   action_dim=sp["train"][0]["actions"].shape[-1],
                   physical_dim=24, hidden_dim=HIDDEN, action_steps=B.C_PREFIX,
                   outcome_dim=NB + NC)
    torch.manual_seed(SEED)
    action_model = M0Predictor(cfg, use_action=True).to(dev)
    torch.manual_seed(SEED)                     # identical initialization
    state_model = M0Predictor(cfg, use_action=False).to(dev)
    models = {"action_conditioned": action_model, "state_only": state_model}
    opts = {k: torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY) for k, m in models.items()}

    best = {k: (float("inf"), None, -1) for k in models}
    curve = a.output / "metrics.jsonl"; curve.write_text("")
    for ep in range(a.epochs):
        order = list(range(len(sp["train"])))
        random.Random(SEED + ep).shuffle(order)
        for m in models.values():
            m.train()
        for i in order:
            for k, m in models.items():
                opts[k].zero_grad(set_to_none=True)
                loss, _, _ = anchor_loss(m, sp["train"][i], st)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {k} loss at epoch {ep}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), MAX_GRAD_NORM)
                opts[k].step()
        row = {"epoch": ep}
        for k, m in models.items():
            v = mean_loss(m, sp["val"], st)
            row[f"{k}_val"] = v
            row[f"{k}_train"] = mean_loss(m, sp["train"], st)
            if v < best[k][0]:
                best[k] = (v, {kk: vv.detach().cpu().clone()
                               for kk, vv in m.state_dict().items()}, ep)
        with curve.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"epoch={ep:02d} action_val={row['action_conditioned_val']:.6f} "
              f"state_val={row['state_only_val']:.6f}", flush=True)

    for k, m in models.items():
        m.load_state_dict(best[k][1]); m.eval()
    torch.save({"config": cfg.__dict__,
                "action_conditioned": best["action_conditioned"][1],
                "state_only": best["state_only"][1],
                "stats": {kk: vv.cpu() for kk, vv in st.items()},
                "best_epoch": {k: best[k][2] for k in best}}, a.output / "best.pt")

    # ---- open the sealed test split, once --------------------------------
    rows = [evaluate(models[k], sp["test"], st, k) for k in models]
    rows += baselines(sp["train"], sp["test"])
    shuffled = torch.stack([sp["test"][0]["actions"][torch.randperm(
        sp["test"][0]["actions"].shape[0], generator=torch.Generator().manual_seed(SEED))]
        for _ in range(1)])[0]
    gaps = {}
    for k, m in models.items():
        base = sum(float(anchor_loss(m, x, st)[0]) for x in sp["test"])
        perm = 0.0
        for x in sp["test"]:
            idx = torch.randperm(x["actions"].shape[0],
                                 generator=torch.Generator().manual_seed(SEED))
            perm += float(anchor_loss(m, x, st, actions=x["actions"][idx])[0])
        gaps[k] = (perm - base) / len(sp["test"])

    summary = {"action": "2M.4", "utc": datetime.now(timezone.utc).isoformat(),
               "groups_path": str(a.groups), "groups_sha256": sha256_file(a.groups),
               "splits": {k: len(v) for k, v in sp.items()},
               "best_epoch": {k: best[k][2] for k in best},
               "primary_target": B.PRIMARY_TARGET,
               "loss": {"physical": W_PHYSICAL, "task": W_TASK, "paired": W_PAIRED},
               "held_out": rows, "action_shuffle_gap": gaps,
               "device": str(dev), "epochs": a.epochs}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== M0.2 held out (test opened once) ===")
    keys = ("physical_mse", "effect_mae", "effect_rmse", "paired_effect_abs_err",
            "top1_regret", "effect_corr")
    print(f"{'model':22s}" + "".join(f"{k:>22s}" for k in keys))
    for r in rows:
        print(f"{r['model']:22s}" + "".join(
            (f"{r[k]:22.6f}" if k in r else f"{'-':>22s}") for k in keys))
    print("\naction-shuffle loss gap:", json.dumps(gaps))
    print("best epochs:", best["action_conditioned"][2], best["state_only"][2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
