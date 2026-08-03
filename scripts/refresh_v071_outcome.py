#!/usr/bin/env python
"""The ONE fixed scorer refresh (pre-registered route, 2026-08-03.md).

lc_full_selected stays byte-frozen. Same K=5 768-256-256-9 ensemble,
same heads/target transforms/member seeds/optimizer/epoch budget/
tolerances/abstention rule, trained FROM SCRATCH. Only the replay rows
change: old union train rows (166) + the complete V7.1.4B r1 bank
(18 anchors x 12 policy candidates, 426 continuation labels, winners/
losers/ties/nulls/regressions all retained). No threshold change, no
LCWM update, no new Q network. Output:
results/libero_loho_public_v1/2026-08-03_v071_outcome_refresh_r1/
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v06_model import V06State  # noqa: E402
from scripts.train_v071_3_outcome import (Head, STATE_VARIANTS,  # noqa: E402
                                          hget, target_vec)

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
UNION = RESULTS / "2026-08-02_v071_union_f1"
LCWM2C = RESULTS / "2026-08-02_v071_lcwm2c_r1"
SEL_R1 = RESULTS / "2026-08-03_v071_selector_r1"
OUT = RESULTS / "2026-08-03_v071_outcome_refresh_r1"
K_ENS, EPOCHS, LR, WD = 5, 300, 1e-3, 1e-4


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.sampler import prefix_forward
    from lcwm.seq_prefix_cache import normalize_actions
    from lcwm.v067_lineage import sha256_file

    device = torch.device("cuda")
    OUT.mkdir(exist_ok=True)
    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]
    union = json.loads((UNION / "union_manifest.json").read_text())
    all_rows = {r["transition_id"]: r for r in union["rows"]}
    roots = {k: Path(v) for k, v in union["replay_roots"].items()}
    sources_rows = {sid: torch.load(b["path"], weights_only=False)
                    ["rows"] for sid, b in
                    union["source_histories"].items()}
    wm = V06State().to(device)
    wm.load_state_dict(torch.load(
        LCWM2C / "checkpoints" / "lc_full_selected.pt",
        weights_only=False)["model"])
    wm.eval()

    def norm_chunk(a):
        ae = torch.as_tensor(np.asarray(a)).float()
        cn = normalize_actions(ae, ref_mean, ref_std)[None].to(device)
        am = (torch.arange(10, device=device)[None] < ae.shape[0])
        if cn.shape[1] < 10:
            cn = torch.cat([cn, torch.zeros(
                1, 10 - cn.shape[1], 7, device=device)], dim=1)
        return cn, am

    X_list, Y_list = [], []
    # ---- old union train rows (cached h; identical to r1 scorer) -------
    from collections import defaultdict
    by_shard = defaultdict(list)
    for r in union["rows"]:
        by_shard[(r["run"], r["shard"])].append(r["transition_id"])
    dirs = STATE_VARIANTS["lc_full"]
    lru, z_cache = {}, {}
    n_old = 0
    for (run, shard), tids in sorted(by_shard.items()):
        s = torch.load(roots[run] / "shards" / shard,
                       weights_only=False)
        for tr in s["transitions"]:
            r = all_rows.get(tr["transition_id"])
            if not r or r["audit_only"] or r["split"] != "train" \
                    or not tr["continuations"]:
                continue
            akey = (r["source_id"], r["decision"])
            if akey not in z_cache:
                sid, d = akey
                z = None
                for dd in range(d + 1):
                    h, m = hget(dirs, lru, f"{sid}__dec{dd}", device)
                    z = wm.initial_state(h, m) if z is None else \
                        wm.step(z, sources_rows[sid][dd - 1]
                                ["chunk_norm"][None, :10].float()
                                .to(device), h, m,
                                action_mask=(torch.arange(
                                    10, device=device)[None]
                                    < sources_rows[sid][dd - 1]
                                    ["executed_len"]))
                z_cache[akey] = z
            z = z_cache[akey]
            cn, am = norm_chunk(tr["actions_env"])
            zt = wm.predict(z, cn, action_mask=am)
            x = torch.cat([z.mean(dim=1)[0],
                           zt.mean(dim=1)[0]]).cpu().numpy()
            for c in tr["continuations"]:
                X_list.append(x)
                Y_list.append(target_vec(c))
            n_old += 1
    print(f"[refresh] old union train transitions: {n_old}",
          flush=True)

    # ---- new r1 bank rows (fresh prefix computes) ----------------------
    runner = Pi05Runner(suite_name="libero_10")
    n_new = 0
    for p in sorted((SEL_R1 / "shards").glob("*.pt")):
        s = torch.load(p, weights_only=False)
        src = torch.load(SEL_R1 / "prospective_sources"
                         / f"{s['source_id']}.pt", weights_only=False)
        instr = src["language_canonical"]
        d = s["decision"]
        z = None
        for dd in range(d + 1):
            rr = src["rows"][dd]
            b = runner._obs_to_policy_batch(rr["obs"], instr)
            p2 = prefix_forward(runner.policy, b)
            h, m = p2.hidden.float(), p2.pad_masks.bool()
            if z is None:
                z = wm.initial_state(h, m)
            else:
                prev = src["rows"][dd - 1]
                z = wm.step(z, prev["chunk_norm"][None, :10].float()
                            .to(device), h, m,
                            action_mask=(torch.arange(
                                10, device=device)[None]
                                < prev["executed_len"]))
        for tr in s["transitions"]:
            if tr["kind"] == "audit" or not tr["continuations"]:
                continue
            cn, am = norm_chunk(tr["actions_env"])
            zt = wm.predict(z, cn, action_mask=am)
            x = torch.cat([z.mean(dim=1)[0],
                           zt.mean(dim=1)[0]]).cpu().numpy()
            for c in tr["continuations"]:
                q = c["q_at_horizons"]
                c2 = {**c, "q_at_horizons":
                      {int(k): v for k, v in q.items()}}
                X_list.append(x)
                Y_list.append(target_vec(c2))
            n_new += 1
        print(f"[refresh] {s['anchor']}: +{12}", flush=True)
    del runner
    torch.cuda.empty_cache()
    print(f"[refresh] new bank transitions: {n_new}; total label "
          f"rows: {len(Y_list)}", flush=True)

    X = torch.tensor(np.stack(X_list)).to(device)
    Y = torch.tensor(np.stack(Y_list)).to(device)
    ens = []
    with torch.enable_grad():
        for k in range(K_ENS):
            torch.manual_seed(k)
            head = Head(X.shape[1]).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=LR,
                                    weight_decay=WD)
            for ep in range(EPOCHS):
                opt.zero_grad(set_to_none=True)
                out = head(X)
                loss = (nn.functional
                        .binary_cross_entropy_with_logits(
                            out[:, 0], Y[:, 0])
                        + nn.functional.mse_loss(out[:, 1:],
                                                 Y[:, 1:]))
                loss.backward()
                opt.step()
            head.eval()
            ens.append(head)
            print(f"[refresh] member {k} final loss "
                  f"{float(loss):.4f}", flush=True)
    torch.save({"lc_full": [h.state_dict() for h in ens]},
               OUT / "scorer_ensembles.pt")
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    (OUT / "run_manifest.json").write_text(json.dumps({
        "schema": "v071_outcome_refresh_manifest_v1",
        "run_schema": "v071", "git_sha": git_sha,
        "refresh": "1 of 1 (pre-registered route; no second refresh)",
        "lcwm_frozen": sha256_file(
            LCWM2C / "checkpoints" / "lc_full_selected.pt"),
        "config": {"k_ens": K_ENS, "epochs": EPOCHS, "lr": LR,
                   "wd": WD, "from_scratch": True,
                   "selection": "NONE"},
        "rows": {"old_train_transitions": n_old,
                 "new_bank_transitions": n_new,
                 "label_rows": len(Y_list)},
        "r1_bank_sha": {
            "outcome_ledger": sha256_file(
                SEL_R1 / "outcome_ledger.jsonl"),
            "seal": sha256_file(SEL_R1 / "ledger_seal.json")},
    }, indent=2))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
