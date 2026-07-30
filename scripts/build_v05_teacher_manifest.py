#!/usr/bin/env python
"""H7.4 step 1 — frozen teacher_state_manifest over the 10 train sources.

Deterministic replay of every H7.2 train rollout (collection noise contract;
cur_wm sources use the frozen v0.4 current-mode adapter, exactly as in
collection/recompute — replay determinism asserted at the collection
snapshot decisions). Along each rollout, with the FROZEN v0.5 checkpoint:

- per decision: token states c / w_reset / g_reset / w_recurrent /
  g_recurrent (reset = w0/g0 + one null-action update, the WM training
  distribution; recurrent = carried with the executed first-ten actions),
  q, executed chunk, first unresolved subgoal;
- ~20 registered teacher decisions per source (20 even bins, median
  unresolved decision per bin): pre-decision raw observation + N=4 fresh
  full-prompt stock candidate pools, seed = noise_seed(d) + 100000*cand
  (candidate 0 = exchangeable stock reference; for stock sources it must
  reproduce the executed chunk);
- rehearsal observations: stock sources, decisions in the first 2/3.

No candidate is executed in the environment (H7.4: no new grounding batch).
Output: datasets/libero_loho_public_v1/teacher_manifest/<source>.pt
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1")
OUT = DATA / "teacher_manifest"
WM_CKPT = (REPO_ROOT / "results" / "libero_loho_public_v1" / "v05_gated_wm"
           / "checkpoint_final.pt")
NOISE_BASE = 30_000_000
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]
TEACHER_BINS = 20
REHEARSAL_FRACTION = 2 / 3
CAND_ATOL = 5e-3


def obs_q(obs) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([
        obs["robot_state"]["eef"]["pos"],
        obs["robot_state"]["eef"]["quat"],
        obs["robot_state"]["gripper"]["qpos"],
    ])).float()


@torch.no_grad()
def h_early_of(runner, batch, pool_image_tokens) -> torch.Tensor:
    images, img_masks = runner.policy._preprocess_images(batch)
    model = runner.policy.model
    sig = torch.cat(
        [model.paligemma_with_expert.embed_image(img)[0].float()
         for img, m in zip(images, img_masks) if bool(m[0])], dim=0)
    return pool_image_tokens(sig)


def select_teacher_decisions(rows) -> list[int]:
    """20 even bins over decisions; median unresolved decision per bin."""
    n = len(rows)
    chosen = []
    for k in range(TEACHER_BINS):
        lo = (k * n) // TEACHER_BINS
        hi = max(((k + 1) * n) // TEACHER_BINS, lo + 1)
        pool = [d for d in range(lo, hi)
                if rows[d]["first_unresolved"] is not None]
        if pool:
            chosen.append(pool[len(pool) // 2])
    return chosen


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import load_lc_adapter, sample_chunks_lc
    from lcwm.loho_public import (SubgoalTracker, load_public_tasks,
                                  make_public_env)
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.taps import pool_image_tokens
    from lcwm.v05_model import V05State

    device = torch.device("cuda")
    model = V05State().to(device)
    bundle = torch.load(WM_CKPT, weights_only=False)
    model.load_state_dict(bundle["model"])
    model.eval()
    wm_hash = hashlib.sha256(WM_CKPT.read_bytes()).hexdigest()

    runner = Pi05Runner(suite_name="libero_10")
    cur_wm_lc, _ = load_lc_adapter(
        REPO_ROOT / "results" / "lc_flow_v04_e2e_v1_cur_wm"
        / "eval_adapter", device="cuda")
    tasks = load_public_tasks()
    OUT.mkdir(parents=True, exist_ok=True)

    summary_path = OUT / "manifest_summary.json"
    summary = (json.loads(summary_path.read_text())
               if summary_path.exists() else
               {"schema": "v05_teacher_manifest_v1",
                "wm_checkpoint_sha256": wm_hash, "sources": {}})
    assert summary["wm_checkpoint_sha256"] == wm_hash

    for task_name in TASK_ORDER:
        task_index = TASK_ORDER.index(task_name)
        subgoals = tasks[task_name]["ordered_subgoals"]
        for policy_name, seed in (("stock", 1400), ("cur_wm", 1420)):
            source_id = f"{task_name}_{policy_name}_s{seed}"
            out_path = OUT / f"{source_id}.pt"
            if out_path.exists():
                print(f"[skip] {source_id}", flush=True)
                continue
            record = torch.load(DATA / f"{source_id}.pt",
                                weights_only=False)
            snap_chunks = {s["decision"]: s["branches"][0]["chunk_norm"]
                           for s in record["snapshots"]}

            def noise_seed(decision):
                return (NOISE_BASE + task_index * 2_000_000
                        + seed * 1_000 + decision)

            env = make_public_env(task_name, EPISODE_LENGTH[task_name])
            try:
                runner.reset()
                obs, _ = env.reset(seed=seed)
                tracker = SubgoalTracker(list(subgoals))
                tracker.start(env)
                instruction = env.task_description

                zero_a = torch.zeros(1, 10, 7, device=device)
                zero_m = torch.zeros(1, 10, dtype=torch.bool,
                                     device=device)
                w_rec = g_rec = a_prev = m_prev = None
                rows, pre_obs = [], []
                t, decision = 0, 0
                term = trunc = False
                while t < env.episode_length:
                    obs_now = copy.deepcopy(obs)
                    batch = runner._obs_to_policy_batch(obs, instruction)
                    prefix = prefix_forward(runner.policy, batch)
                    if policy_name == "cur_wm":
                        z = cur_wm_lc.posterior(
                            prefix.hidden, prefix.pad_masks)
                        bias = cur_wm_lc.adarms_bias(z)
                        chunk = sample_chunks_lc(
                            runner.policy, batch, bias, n=1,
                            seed=noise_seed(decision), prefix=prefix)
                    else:
                        chunk = sample_chunks(
                            runner.policy, batch, n=1,
                            seed=noise_seed(decision), prefix=prefix)
                    if decision in snap_chunks:
                        err = float((chunk[0].float().cpu()
                                     - snap_chunks[decision]).abs().max())
                        assert err < 1e-4, (
                            f"{source_id} d={decision}: replay diverges "
                            f"({err:.2e})")
                    fu = next((i for i in range(len(subgoals))
                               if i not in tracker.completed), None)

                    h_early = h_early_of(
                        runner, batch, pool_image_tokens)[None].float()
                    h_late = prefix.hidden.float()
                    mask = prefix.pad_masks.bool()
                    q = obs_q(obs)[None].to(device)
                    w0, g0 = model.initial(1, device)
                    w_r = model.step_physical(
                        w0, zero_a, h_early, q, action_mask=zero_m)
                    c = model.current(h_early, h_late, mask, q)
                    g_r = model.step_task(
                        g0, w_r, zero_a, h_late, mask, action_mask=zero_m)
                    if w_rec is None:
                        w_now, g_now = w_r, g_r
                    else:
                        w_now = model.step_physical(
                            w_rec, a_prev, h_early, q, action_mask=m_prev)
                        g_now = model.step_task(
                            g_rec, w_now, a_prev, h_late, mask,
                            action_mask=m_prev)

                    executed = 0
                    for a in runner.chunk_to_env(chunk[:, :10]):
                        obs, _r, term, trunc, _i = env.step(a)
                        t += 1
                        executed += 1
                        if term or trunc:
                            break
                    tracker.update(env, t)

                    rows.append({
                        "decision": decision, "t": t - executed,
                        "first_unresolved": fu,
                        "executed_len": executed,
                        "chunk_norm": chunk[0].float().cpu(),
                        "q": q[0].cpu(),
                        "c": c[0].cpu(), "w_reset": w_r[0].cpu(),
                        "g_reset": g_r[0].cpu(),
                        "w_rec": w_now[0].cpu(),
                        "g_rec": g_now[0].cpu(),
                    })
                    pre_obs.append(obs_now)
                    a_prev = chunk[:, :10].float()
                    m_prev = (torch.arange(10, device=device)[None]
                              < executed)
                    w_rec, g_rec = w_now, g_now
                    decision += 1
                    if term or trunc:
                        break

                n = len(rows)
                teacher_ds = select_teacher_decisions(rows)
                rehearsal_ds = ([d for d in range(n)
                                 if d < REHEARSAL_FRACTION * n]
                                if policy_name == "stock" else [])

                cand0_err = 0.0
                for d in teacher_ds:
                    batch = runner._obs_to_policy_batch(
                        pre_obs[d], instruction)
                    prefix = prefix_forward(runner.policy, batch)
                    cands, seeds = [], []
                    for cand in range(4):
                        s = noise_seed(d) + 100_000 * cand
                        cands.append(sample_chunks(
                            runner.policy, batch, n=1, seed=s,
                            prefix=prefix)[0].float().cpu())
                        seeds.append(s)
                    if policy_name == "stock":
                        err = float((cands[0]
                                     - rows[d]["chunk_norm"]).abs().max())
                        cand0_err = max(cand0_err, err)
                        assert err < CAND_ATOL, (
                            f"{source_id} d={d}: stock candidate-0 differs "
                            f"from executed chunk ({err:.2e})")
                    rows[d]["candidates"] = torch.stack(cands)
                    rows[d]["candidate_seeds"] = seeds
                for d in sorted(set(teacher_ds) | set(rehearsal_ds)):
                    rows[d]["obs"] = pre_obs[d]

                div = torch.tensor([
                    float((r["w_rec"] - r["w_reset"]).norm()
                          / r["w_reset"].norm().clamp_min(1e-8))
                    for r in rows])
                torch.save({
                    "schema": "v05_teacher_manifest_v1",
                    "source_id": source_id, "task": task_name,
                    "policy": policy_name, "seed": seed,
                    "split": record["split"],
                    "language_canonical": instruction,
                    "n_subgoals": len(subgoals),
                    "wm_checkpoint_sha256": wm_hash,
                    "teacher_decisions": teacher_ds,
                    "rehearsal_decisions": rehearsal_ds,
                    "rows": rows,
                }, out_path)
                summary["sources"][source_id] = {
                    "n_decisions": n,
                    "n_teacher": len(teacher_ds),
                    "n_rehearsal": len(rehearsal_ds),
                    "terminated": bool(term or trunc),
                    "stock_cand0_max_err": cand0_err,
                    "w_rec_vs_reset_reldiv_mean": float(div.mean()),
                    "w_rec_vs_reset_reldiv_max": float(div.max()),
                }
                summary_path.write_text(json.dumps(summary, indent=2))
                print(f"[manifest] {source_id}: {n} decisions, "
                      f"{len(teacher_ds)} teacher, "
                      f"{len(rehearsal_ds)} rehearsal, "
                      f"cand0_err={cand0_err:.1e}, "
                      f"w_div={float(div.mean()):.3f}", flush=True)
            finally:
                env.close()
    print("teacher manifest complete", flush=True)


if __name__ == "__main__":
    main()
