#!/usr/bin/env python
"""H7.4 step 2 — score the manifest candidate pools with the FROZEN v0.5
checkpoint and freeze the teacher table for all four arms.

Registered scoring (H7.4 scoping, 2026-07-30):
- state = reset-computed (c, w_reset, g_reset) — the WM training
  distribution — for every arm, so reset/recurrent share identical
  selections;
- Ĝ_i = v̂_i (predicted continuation Q_public). Deviation registered: no
  P̂(terminal success) term (0/160 successes in training continuations);
- select the best non-stock candidate iff Ĝ_best > Ĝ_stock (candidate 0);
- fixed-random control: generator seed 777000 + state_index (states ordered
  by (source_id, decision) over ALL teacher states), uniform over {1,2,3},
  used only at the shared positive-state mask.

Also computes the registered non-gating diagnostic: within-sibling rank
agreement (Spearman) of frozen v̂ against OBSERVED continuation Q_public on
the grounded H7.2 branch groups, train and dev separately.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.v05_model import V05State  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1")
MANIFEST = DATA / "teacher_manifest"
WM_CKPT = (REPO_ROOT / "results" / "libero_loho_public_v1" / "v05_gated_wm_r1"
           / "checkpoint_final.pt")
OUT = REPO_ROOT / "results" / "libero_loho_public_v1" / "v05_teachers"
RANDOM_SEED_BASE = 777_000


@torch.no_grad()
def score_pool(model, c, w_reset, g_reset, chunks, device):
    """v̂/dq̂/r̂ for every candidate in one batch. chunks: [N,50,7]."""
    n = chunks.shape[0]
    prior = model.physical_prior(
        w_reset.expand(n, -1, -1), chunks[:, :10].to(device))
    out = model.outcomes(prior, c.expand(n, -1, -1),
                         g_reset.expand(n, -1, -1))
    return {k: out[k].float().cpu() for k in
            ("v_hat", "dq_public_hat", "r_hat")}


@torch.no_grad()
def snapshot_state(model, feats, device):
    """Reset-computed (c, w, g) from a feature sidecar state — the exact
    snapshot_loss recipe of the H7.3 trainer."""
    h_early = feats["h_early"][None].float().to(device)
    h_late = feats["h_late"][None].float().to(device)
    mask = feats["h_late_mask"][None].to(device)
    q = feats["q"][None].to(device)
    zero_a = torch.zeros(1, 10, 7, device=device)
    zero_m = torch.zeros(1, 10, dtype=torch.bool, device=device)
    w0, g0 = model.initial(1, device)
    w = model.step_physical(w0, zero_a, h_early, q, action_mask=zero_m)
    c = model.current(h_early, h_late, mask, q)
    g = model.step_task(g0, w, zero_a, h_late, mask, action_mask=zero_m)
    return c, w, g


def spearman(a: list[float], b: list[float]) -> float | None:
    ta, tb = torch.tensor(a), torch.tensor(b)
    if ta.std() == 0 or tb.std() == 0:
        return None

    def ranks(x):
        order = x.argsort()
        r = torch.empty_like(x)
        r[order] = torch.arange(len(x), dtype=x.dtype)
        # average ties
        for v in x.unique():
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = ranks(ta), ranks(tb)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum()
                 / (ra.norm() * rb.norm()).clamp_min(1e-12))


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda")
    model = V05State().to(device)
    bundle = torch.load(WM_CKPT, weights_only=False)
    model.load_state_dict(bundle["model"])
    model.eval()
    wm_hash = hashlib.sha256(WM_CKPT.read_bytes()).hexdigest()

    # ---- teacher table over manifest pools --------------------------------
    rows = []
    state_index = 0
    for path in sorted(MANIFEST.glob("loho_*.pt")):
        m = torch.load(path, weights_only=False)
        assert m["wm_checkpoint_sha256"] == wm_hash
        for d in m["teacher_decisions"]:
            row = m["rows"][d]
            c = row["c"][None].to(device)
            w = row["w_reset"][None].to(device)
            g = row["g_reset"][None].to(device)
            cands = row["candidates"]
            scores = score_pool(model, c, w, g, cands, device)
            g_hat = scores["v_hat"]
            best = int(g_hat[1:].argmax()) + 1
            advantage = float(g_hat[best] - g_hat[0])
            selected = best if advantage > 0 else 0
            gen = torch.Generator().manual_seed(
                RANDOM_SEED_BASE + state_index)
            random_selected = int(
                torch.randint(1, 4, (1,), generator=gen))
            rows.append({
                "source_id": m["source_id"], "task": m["task"],
                "source_policy": m["policy"], "decision": d,
                "first_unresolved": row["first_unresolved"],
                "candidate_seeds": row["candidate_seeds"],
                "g_hat": g_hat, "dq_public_hat": scores["dq_public_hat"],
                "r_hat": scores["r_hat"],
                "selected": selected,
                "advantage": advantage if selected else 0.0,
                "random_selected": random_selected,
                "candidate_sha256": [
                    hashlib.sha256(
                        cands[i].numpy().tobytes()).hexdigest()[:16]
                    for i in range(cands.shape[0])],
                "diversity_l2": float(
                    (cands[1:, :10] - cands[0:1, :10])
                    .norm(dim=-1).mean()),
            })
            state_index += 1
        print(f"[score] {m['source_id']}: "
              f"{len(m['teacher_decisions'])} states", flush=True)

    positive = [r for r in rows if r["selected"] != 0]
    by_phase: dict[str, list] = {}
    for r in rows:
        by_phase.setdefault(str(r["first_unresolved"]), []).append(r)

    # ---- grounded rank-agreement diagnostic (train + dev) -----------------
    rank_diag = {}
    for split_want in ("train", "dev"):
        rhos, argmax_hits, skipped = [], [], 0
        for rec_path in sorted(DATA.glob("loho_*.pt")):
            if rec_path.name.endswith(".features.pt"):
                continue
            record = torch.load(rec_path, weights_only=False)
            if record["split"] != split_want:
                continue
            feats_all = torch.load(
                rec_path.with_suffix(".features.pt"), weights_only=False)
            for snap in record["snapshots"]:
                feats = feats_all["snapshots"].get(snap["decision"])
                if feats is None:
                    continue
                c, w, g = snapshot_state(model, feats["state"], device)
                chunks = torch.stack(
                    [b["chunk_norm"] for b in snap["branches"]])
                pred = score_pool(model, c, w, g, chunks, device)
                v_pred = pred["v_hat"].tolist()
                v_obs = [b["continuation"]["q_public"]
                         for b in snap["branches"]]
                rho = spearman(v_pred, v_obs)
                if rho is None:
                    skipped += 1
                    continue
                rhos.append(rho)
                obs_t = torch.tensor(v_obs)
                argmax_hits.append(float(
                    obs_t[int(torch.tensor(v_pred).argmax())]
                    == obs_t.max()))
        rank_diag[split_want] = {
            "n_groups": len(rhos),
            "n_skipped_zero_variance": skipped,
            "spearman_mean": (float(torch.tensor(rhos).mean())
                              if rhos else None),
            "argmax_hit_rate": (float(torch.tensor(argmax_hits).mean())
                                if argmax_hits else None),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "v05_candidate_teacher_v1", "rows": rows,
                "wm_checkpoint_sha256": wm_hash}, OUT / "teachers.pt")
    diagnostics = {
        "n_teacher_states": len(rows),
        "wm_checkpoint_sha256": wm_hash,
        "positive_fraction": len(positive) / max(len(rows), 1),
        "positive_by_source": {
            s: sum(1 for r in positive if r["source_id"] == s)
            / max(sum(1 for r in rows if r["source_id"] == s), 1)
            for s in sorted({r["source_id"] for r in rows})},
        "positive_by_phase": {
            k: sum(1 for r in v if r["selected"] != 0) / len(v)
            for k, v in sorted(by_phase.items())},
        "advantage_quantiles": (
            torch.tensor([r["advantage"] for r in positive])
            .quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()
            if positive else None),
        "mean_candidate_diversity_l2": float(torch.tensor(
            [r["diversity_l2"] for r in rows]).mean()),
        "grounded_rank_agreement": rank_diag,
    }
    (OUT / "teacher_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"-> {OUT / 'teachers.pt'}", flush=True)


if __name__ == "__main__":
    main()
