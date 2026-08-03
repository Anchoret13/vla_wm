#!/usr/bin/env python
"""V7.1.4A unit test — compiled selector assertions.

Registered as a UNIT TEST, not new scientific evidence: synthetic
invariant checks + one tensor-only replay of the existing V7.1.3 dev
rows to verify the compiled selector obeys the frozen assertions.
Writes 2026-08-03_v071_selector_r1/analysis/selector_unit_test.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v071_selector import (MASKED, SELECT_ORDER, project,  # noqa: E402
                                select)
from lcwm.v06_model import V06State  # noqa: E402
from lcwm.v071_loader import V071Loader  # noqa: E402
from scripts.train_v071_3_outcome import (Head, STATE_VARIANTS,  # noqa: E402
                                          hget, target_vec)

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
UNION = RESULTS / "2026-08-02_v071_union_f1"
OUTCOME = RESULTS / "2026-08-02_v071_outcome_r1"
LCWM2C = RESULTS / "2026-08-02_v071_lcwm2c_r1"
ROOT = RESULTS / "2026-08-03_v071_selector_r1"


def synthetic_checks():
    K = 5
    base = np.tile(np.array([0.0, .1, .2, .3, .4, 0.5, 0.5,
                             0.0, -0.5]), (K, 1))
    # 1) masked-only difference -> tied_ineligible, decisive None
    cand = base.copy()
    cand[:, 0] += 0.9          # success head (masked)
    cand[:, 7] -= 0.3          # damage head (masked)
    r = select({"u0": base, "c1": cand}, "u0", run_id="t",
               source_id="s", decision=0)
    assert r["selected"] == "u0"
    assert r["candidates"]["c1"]["eligibility"] == "tied_ineligible"
    assert r["candidates"]["c1"]["decisive_component"] is None
    # 2) uncertainty crossing the boundary -> ineligible_uncertain
    cand = base.copy()
    cand[:, 6] += np.array([0.5, -0.5, 0.5, -0.5, 0.5])
    r = select({"u0": base, "c1": cand}, "u0", run_id="t",
               source_id="s", decision=0)
    assert r["candidates"]["c1"]["eligibility"] \
        == "ineligible_uncertain"
    # 3) clear confidence-positive on p_valid -> selected
    cand = base.copy()
    cand[:, 6] += 0.3
    r = select({"u0": base, "c1": cand}, "u0", run_id="t",
               source_id="s", decision=0)
    assert r["selected"] == "c1" \
        and r["candidates"]["c1"]["decisive_component"] \
        == "p_valid_100"
    # 4) confidence-negative -> ineligible
    cand = base.copy()
    cand[:, 6] -= 0.3
    r = select({"u0": base, "c1": cand}, "u0", run_id="t",
               source_id="s", decision=0)
    assert r["candidates"]["c1"]["eligibility"] \
        == "ineligible_negative" and r["selected"] == "u0"
    # 5) two identical positives -> deterministic SHA tie-break
    c1, c2 = base.copy(), base.copy()
    c1[:, 6] += 0.3
    c2[:, 6] += 0.3
    r1 = select({"u0": base, "c1": c1, "c2": c2}, "u0", run_id="t",
                source_id="s", decision=0)
    r2 = select({"u0": base, "c1": c1, "c2": c2}, "u0", run_id="t",
                source_id="s", decision=0)
    assert r1["selected"] == r2["selected"] \
        and r1["selection_reason"] == "sha_tie_break"
    # 6) projection: out-of-domain output cannot create headroom
    cand = base.copy()
    cand[:, 6] = 3.0           # raw 3.0 projects to 1.0
    stock = base.copy()
    stock[:, 6] = 1.0
    r = select({"u0": stock, "c1": cand}, "u0", run_id="t",
               source_id="s", decision=0)
    assert r["candidates"]["c1"]["eligibility"] != \
        "confidence_positive", "projection failed to cap headroom"
    assert float(project(cand[0])[6]) == 1.0
    return 6


@torch.no_grad()
def real_row_replay():
    """Tensor-only replay of the existing dev-pool rows."""
    from lcwm.seq_prefix_cache import normalize_actions
    device = torch.device("cuda")
    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]
    loader = V071Loader(UNION)
    union = json.loads((UNION / "union_manifest.json").read_text())
    roots = {k: Path(v) for k, v in union["replay_roots"].items()}
    all_rows = {r["transition_id"]: r for r in union["rows"]}
    sources_rows = {sid: torch.load(b["path"], weights_only=False)
                    ["rows"] for sid, b in
                    union["source_histories"].items()}
    dev_anchors = ["loho_t2_basket3_correction_s2112_d75",
                   "loho_t3_tray_correction_s2122_d2",
                   "loho_t5_drawer_cabinet_correction_s2142_d2"]
    labeled = {}
    for (run, shard) in sorted({(r["run"], r["shard"])
                                for r in all_rows.values()
                                if r["anchor"] in dev_anchors}):
        s = torch.load(roots[run] / "shards" / shard,
                       weights_only=False)
        for tr in s["transitions"]:
            r = all_rows.get(tr["transition_id"])
            if r and r["anchor"] in dev_anchors \
                    and tr["continuations"]:
                labeled[tr["transition_id"]] = (r, tr)
    model = V06State().to(device)
    st = torch.load(LCWM2C / "checkpoints" / "lc_full_selected.pt",
                    weights_only=False)
    model.load_state_dict(st["model"])
    model.eval()
    ens_sd = torch.load(OUTCOME / "scorer_ensembles.pt",
                        weights_only=False)["lc_full"]
    heads = []
    for sd in ens_sd:
        h = Head(768).to(device)
        h.load_state_dict(sd)
        h.eval()
        heads.append(h)
    dirs = STATE_VARIANTS["lc_full"]
    lru = {}
    ledgers = []
    for _pass in range(2):          # byte-determinism check
        rows_out = []
        for anchor in dev_anchors:
            tids = sorted(t for t, (r, _tr) in labeled.items()
                          if r["anchor"] == anchor)
            sid = all_rows[tids[0]]["source_id"]
            d = all_rows[tids[0]]["decision"]
            z = None
            for dd in range(d + 1):
                h, m = hget(dirs, lru, f"{sid}__dec{dd}", device)
                if z is None:
                    z = model.initial_state(h, m)
                else:
                    prev = sources_rows[sid][dd - 1]
                    a = prev["chunk_norm"][None, :10].float() \
                        .to(device)
                    am = (torch.arange(10, device=device)[None]
                          < prev["executed_len"])
                    z = model.step(z, a, h, m, action_mask=am)
            preds = {}
            for tid in tids:
                _r, tr = labeled[tid]
                ae = torch.from_numpy(
                    np.asarray(tr["actions_env"])).float()
                cn = normalize_actions(ae, ref_mean,
                                       ref_std)[None].to(device)
                am = (torch.arange(10, device=device)[None]
                      < ae.shape[0])
                zt = model.predict(z, cn, action_mask=am)
                x = torch.cat([z.mean(dim=1)[0],
                               zt.mean(dim=1)[0]])[None]
                outs = []
                for head in heads:
                    o = head(x)[0]
                    o = torch.cat([torch.sigmoid(o[:1]), o[1:]])
                    outs.append(o.cpu().numpy())
                preds[tid] = np.stack(outs)
            stock_id = next(t for t in tids if t.endswith("u0_fresh"))
            row = select(preds, stock_id,
                         run_id="v071_selector_unit",
                         source_id=sid, decision=d)
            assert row["candidates"] and all(
                c["decisive_component"] not in MASKED
                for c in row["candidates"].values())
            rows_out.append(row)
        ledgers.append(json.dumps(rows_out, sort_keys=True))
    assert ledgers[0] == ledgers[1], "ledger not byte-deterministic"
    return json.loads(ledgers[0])


def main() -> None:
    n_syn = synthetic_checks()
    real = real_row_replay()
    (ROOT / "analysis").mkdir(parents=True, exist_ok=True)
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    out = {
        "schema": "v071_selector_unit_test_v1",
        "status": "UNIT TEST ONLY — not scientific evidence",
        "git_sha": git_sha,
        "synthetic_assertions_passed": n_syn,
        "select_order": [n for n, _i, _t in SELECT_ORDER],
        "masked": sorted(MASKED),
        "real_dev_rows_replay": real,
        "byte_deterministic": True,
    }
    (ROOT / "analysis" / "selector_unit_test.json").write_text(
        json.dumps(out, indent=2))
    for r in real:
        print(f"{r['source_id']}_d{r['decision']}: "
              f"selected={r['selected'].rsplit('_', 1)[-1]} "
              f"intervention={r['intervention']} "
              f"reason={r['selection_reason']}", flush=True)
    print(f"UNIT TEST PASS ({n_syn} synthetic + 3 real anchors, "
          f"byte-deterministic)", flush=True)


if __name__ == "__main__":
    main()
