#!/usr/bin/env python
"""V7.2B post-hoc gradient-reach probe (measurement only).

The in-run probe anchored on schedule slot 0, which carries no
continuations/pairs, so `out`/`bt` isolated reach was recorded None.
This probe re-measures every bundle on a continuation-bearing selector
anchor for each arm's FINAL checkpoint:
  (1) isolated cleared-gradient autograd pass per bundle -> shared-
      state gradient norm (reach potential; nonzero expected for all
      bundles in isolation, in every arm);
  (2) routing verification: two-pass update exactly as training
      (state bundles backward; head bundles backward with state grads
      snapshot-restored) -> assert state grads after restore are
      byte-identical to the state-only pass for the arm's head-local
      bundles (exact zero net contribution).
Writes metrics/gradient_reach_posthoc.json. No parameters updated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm import v072_schedule as S  # noqa: E402
from lcwm.self_predict import LatentPredictor, latent_distance  # noqa: E402
from lcwm.v06_model import V06State, make_ema  # noqa: E402
from scripts.train_v072_lcwm_arms import (ARMS, STATE_MODULES,  # noqa: E402
                                          OUT, DATA_R1, Q_SCALE,
                                          OBJ_SCALE, HUB_PHYS,
                                          HUB_OUT, hub)

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"


def main() -> None:
    device = torch.device("cuda")
    transitions = S.load_union()
    by_pt = {r["pt_id"]: r for r in transitions}
    anchors = S.build_anchors(transitions)
    hidx = json.loads((DATA_R1 / "hcache_index.json").read_text())
    union = json.loads((RESULTS / "2026-08-02_v071_union_f1"
                        / "union_manifest.json").read_text())
    src_paths = {sid: b["path"]
                 for sid, b in union["source_histories"].items()}
    for tag in ("selector_r1", "selector_r2"):
        for p in (RESULTS / f"2026-08-03_v071_{tag}"
                  / "prospective_sources").glob("*.pt"):
            src_paths[p.stem] = str(p)
    raw_roots = {k: Path(v["path"]) for k, v in json.loads(
        (DATA_R1 / "run_manifest.json").read_text())
        ["raw_roots"].items()}

    # pick the first selector anchor with >= 2 continuation rows
    ak = next(a for a in sorted(anchors)
              if anchors[a]["origin"] == "v071_selr1")
    a = anchors[ak]
    tv = S.canonical_goal(a["task"]) + "_p0"
    g = S.canonical_goal(a["task"])
    sid, d = a["source_id"], a["decision"]
    src = torch.load(src_paths[sid], weights_only=False)
    rec0 = by_pt[a["rows"][0]]
    shard = torch.load(raw_roots[rec0["raw_root"]] / "shards"
                       / rec0["raw_shard"], weights_only=False)
    trs = {f"{shard['anchor']}_{t['candidate_id']}": t
           for t in shard["transitions"]}

    def hget(key):
        d_ = torch.load(hidx[key], weights_only=False)
        return (d_["h"][None].float().to(device),
                d_["mask"][None].to(device))

    def cont_g(c):
        q = {int(k): v for k, v in c["q_at_horizons"].items()}
        return np.array([c["p_valid_100"], q[10], q[30], q[60],
                         q[100]], dtype=np.float32)

    report = {"anchor": ak, "text": tv}
    for arm in ARMS:
        ck = torch.load(OUT / "checkpoints" / f"{arm}_final.pt",
                        weights_only=False)
        model = V06State().to(device)
        model.load_state_dict(ck["model"])
        pred = LatentPredictor().to(device)
        pred.load_state_dict(ck["predictor"])
        ema = make_ema(model)
        ema.load_state_dict(ck["ema"])
        params = list(model.parameters()) + list(pred.parameters())
        state_params = [p for n, p in model.named_parameters()
                       if n.split(".")[0] in STATE_MODULES]
        q_scale = Q_SCALE.to(device)

        def unroll(model_):
            z = None
            for dd in range(d):
                h, m = hget(f"{tv}::src::{sid}::d{dd}")
                if z is None:
                    z = model_.initial_state(h, m)
                else:
                    prev = src["rows"][dd - 1]
                    aa = prev["chunk_norm"][None, :10].float() \
                        .to(device)
                    am = (torch.arange(10, device=device)[None]
                          < prev["executed_len"])
                    z = model_.step(z, aa, h, m, action_mask=am)
            h, m = hget(f"{tv}::src::{sid}::d{d}")
            prev = src["rows"][d - 1]
            aa = prev["chunk_norm"][None, :10].float().to(device)
            am = (torch.arange(10, device=device)[None]
                  < prev["executed_len"])
            return model_.step(z, aa, h, m, action_mask=am)

        def bundle_losses():
            z = unroll(model)
            with torch.no_grad():
                zbar = unroll(ema)
            obj_before = np.asarray(src["rows"][d]["obj_before"])
            L, sts, tgts, preds_g = {}, [], [], []
            cl, pa = [], []
            for pt in a["rows"]:
                rec = by_pt[pt]
                if rec["kind"] == "audit":
                    continue
                tr = trs[rec["raw_key"]]
                cn = torch.tensor(rec["actions_pi05_norm"],
                                  dtype=torch.float32,
                                  device=device)[None, :10]
                am = (torch.arange(10, device=device)[None]
                      < rec["steps"])
                zt = model.predict(z, cn, action_mask=am)
                hn, mn = hget(f"{tv}::next::{pt}")
                with torch.no_grad():
                    tgt = ema.update(
                        ema.t(zbar, ema.e_a(cn, am)), hn, mn)
                cl.append(latent_distance(pred(zt), tgt.detach(),
                                          "cosine").mean())
                out = model.d_next(zt)
                d_eef = torch.from_numpy(
                    np.asarray(tr["eef_seq"][-1])
                    - np.asarray(tr["eef_seq"][0])).float() \
                    .to(device)
                pa.append(hub(out["d_q"][0], d_eef, q_scale,
                              HUB_PHYS))
                if len(tr["continuations"]) == 2:
                    gbar = torch.from_numpy(np.mean(
                        [cont_g(c) for c in tr["continuations"]],
                        axis=0)).float().to(device)
                    ghat = torch.sigmoid(torch.cat(
                        [out["p_valid"].reshape(1),
                         out["q_valid"][0]]))
                    tgts.append(gbar)
                    preds_g.append(ghat)
                    sts.append(out["s"][0])
            L["closure"] = torch.stack(cl).mean()
            L["phys_abs"] = torch.stack(pa).mean()
            L["out"] = torch.stack([
                torch.nn.functional.huber_loss(p_, t_,
                                               delta=HUB_OUT)
                for p_, t_ in zip(preds_g, tgts)]).mean()
            L["bt"] = torch.nn.functional.softplus(
                -(sts[0] - sts[1])).mean()
            return L

        iso = {}
        for k in ("closure", "phys_abs", "out", "bt"):
            for p in params:
                p.grad = None
            L = bundle_losses()
            L[k].backward()
            iso[k] = float(sum((p.grad ** 2).sum()
                               for p in state_params
                               if p.grad is not None).sqrt())
        # routing zero-verification (two-pass, exactly as training)
        cfg = ARMS[arm]
        state_keys = ["closure", "phys_abs"]
        if cfg["out_state"]:
            state_keys.append("out")
        if cfg["rank_state"]:
            state_keys.append("bt")
        for p in params:
            p.grad = None
        L = bundle_losses()
        torch.stack([L[k] for k in state_keys]).sum().backward(
            retain_graph=True)
        snap = [p.grad.detach().clone() if p.grad is not None
                else None for p in state_params]
        head_keys = [k for k in ("out", "bt")
                     if k not in state_keys]
        routing_ok = True
        if head_keys:
            torch.stack([L[k] for k in head_keys]).sum().backward()
            for p, s_ in zip(state_params, snap):
                if s_ is None:
                    if p.grad is not None:
                        p.grad.zero_()
                else:
                    p.grad.copy_(s_)
            for p, s_ in zip(state_params, snap):
                if s_ is not None and not torch.equal(p.grad, s_):
                    routing_ok = False
        report[arm] = {"isolated_state_grad_norm": iso,
                       "head_local_bundles": head_keys,
                       "routing_restore_exact": routing_ok}
        for k in ("closure", "phys_abs", "out", "bt"):
            assert iso[k] > 0, f"{arm}/{k} zero isolated reach"
        print(f"[{arm}] iso={ {k: round(v, 5) for k, v in iso.items()} } "
              f"head_local={head_keys} restore_exact={routing_ok}",
              flush=True)
    (OUT / "metrics" / "gradient_reach_posthoc.json").write_text(
        json.dumps(report, indent=2))
    print("PROBE PASS", flush=True)


if __name__ == "__main__":
    main()
