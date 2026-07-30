#!/usr/bin/env python
"""H7.3 prerequisite — deterministic feature recompute for the H7.2
collection (contract gap recorded in the progress note: the collector kept
replay provenance but not visual features).

For every collected source: replay the SAME rollout (seed + noise stream),
snap at the recorded snapshot decisions, and for the snapshot state and
every stored branch chunk capture:
- full real-prompt prefix hidden/mask (H_late; LC-posterior native input);
- SigLIP pre-trunk image tokens (H_early, pooled 128x2048 as in seq_v2);
- q / object positions before and after each branch's 10 executed actions.
Chunks are NOT resampled — the stored chunks are executed verbatim, and the
replayed source chunk is asserted equal to the stored candidate-0 chunk
(determinism check).
Output: sidecar files <source>.features.pt next to each collection record.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1")
NOISE_BASE = 30_000_000
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]


@torch.no_grad()
def capture_features(runner, env, instruction, pool_image_tokens):
    from lcwm.sampler import prefix_forward
    from lcwm.probe_data import body_positions, discover_object_bodies

    obs = env._format_raw_obs(env._env.env._get_observations())
    batch = runner._obs_to_policy_batch(obs, instruction)
    prefix = prefix_forward(runner.policy, batch)
    images, img_masks = runner.policy._preprocess_images(batch)
    model = runner.policy.model
    early = []
    for img, mask in zip(images, img_masks):
        if bool(mask.all()):
            early.append(model.paligemma_with_expert.embed_image(img)[0])
    early_pooled = pool_image_tokens(torch.cat(early, dim=0)[None])[0]
    bodies = discover_object_bodies(env)
    q = torch.from_numpy(np.concatenate([
        obs["robot_state"]["eef"]["pos"],
        obs["robot_state"]["eef"]["quat"],
        obs["robot_state"]["gripper"]["qpos"],
    ])).float()
    obj = torch.from_numpy(
        body_positions(env, list(bodies.values()))).float()
    return {
        "h_late": prefix.hidden[0].half().cpu(),
        "h_late_mask": prefix.pad_masks[0].bool().cpu(),
        "h_early": early_pooled.half().cpu(),
        "q": q, "obj_pos": obj,
        "object_names": list(bodies),
    }


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import load_lc_adapter, sample_chunks_lc
    from lcwm.loho_public import make_public_env
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.snapshot import snap, restore
    from lcwm.taps import pool_image_tokens

    runner = Pi05Runner(suite_name="libero_10")
    cur_wm_lc, _ = load_lc_adapter(
        REPO_ROOT / "results" / "lc_flow_v04_e2e_v1_cur_wm"
        / "eval_adapter", device="cuda")

    for record_path in sorted(DATA.glob("loho_*.pt")):
        if record_path.name.endswith(".features.pt"):
            continue
        out_path = record_path.with_suffix(".features.pt")
        if out_path.exists():
            print(f"[skip] {out_path.name}", flush=True)
            continue
        record = torch.load(record_path, weights_only=False)
        task_name, seed = record["task"], record["seed"]
        task_index = TASK_ORDER.index(task_name)
        policy_name = record["policy"]
        instruction = record["language"]["canonical"]

        def noise_seed(decision):
            return (NOISE_BASE + task_index * 2_000_000
                    + seed * 1_000 + decision)

        env = make_public_env(task_name, EPISODE_LENGTH[task_name])
        try:
            runner.reset()
            obs, _ = env.reset(seed=seed)
            wanted = {s["decision"]: s for s in record["snapshots"]}
            snap_states = {}
            t, decision = 0, 0
            while t < env.episode_length and len(snap_states) < len(wanted):
                if decision in wanted:
                    snap_states[decision] = snap(
                        env, t=t, suite_name="loho_public", task_id=0)
                batch = runner._obs_to_policy_batch(obs, instruction)
                prefix = prefix_forward(runner.policy, batch)
                if policy_name == "cur_wm":
                    z = cur_wm_lc.posterior(prefix.hidden, prefix.pad_masks)
                    bias = cur_wm_lc.adarms_bias(z)
                    chunk = sample_chunks_lc(
                        runner.policy, batch, bias, n=1,
                        seed=noise_seed(decision), prefix=prefix)
                else:
                    chunk = sample_chunks(
                        runner.policy, batch, n=1,
                        seed=noise_seed(decision), prefix=prefix)
                if decision in wanted:
                    stored = wanted[decision]["branches"][0]["chunk_norm"]
                    err = float((chunk[0].float().cpu() - stored).abs().max())
                    assert err < 1e-4, (
                        f"{record_path.name} d={decision}: replay chunk "
                        f"diverges from stored candidate-0 ({err:.2e})")
                for a in runner.chunk_to_env(chunk[:, :10]):
                    obs, _r, term, trunc, _i = env.step(a)
                    t += 1
                    if term or trunc:
                        break
                decision += 1
                if term or trunc:
                    break

            features = {"schema": "loho_features_v1", "snapshots": {}}
            for d, snapshot in snap_states.items():
                restore(env, snapshot)
                at_snapshot = capture_features(
                    runner, env, instruction, pool_image_tokens)
                branch_feats = []
                for branch in wanted[d]["branches"]:
                    restore(env, snapshot)
                    for a in runner.chunk_to_env(
                            branch["chunk_norm"][None].cuda()[:, :10]):
                        env.step(a)
                    branch_feats.append(capture_features(
                        runner, env, instruction, pool_image_tokens))
                features["snapshots"][d] = {
                    "state": at_snapshot, "after_branches": branch_feats,
                }
            torch.save(features, out_path)
            print(f"[features] {record_path.name}: "
                  f"{len(snap_states)} snapshots x 1+4 captures",
                  flush=True)
        finally:
            env.close()
    print("feature recompute complete", flush=True)


if __name__ == "__main__":
    main()
