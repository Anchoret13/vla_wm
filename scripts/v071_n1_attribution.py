#!/usr/bin/env python
"""V7.1 policy-boundary closure — bounded N1 attribution check.

Registered item (2026-08-02.md, non-blocking): evaluate the EXISTING
V7.1.0B smoke checkpoints once on the same live re-reached
observation/history under `full LC`, `LCProj=0`, and reset/shuffled
recurrent state; persist generated chunk hashes. This is a bounded
attribution measurement — no training, no selection, no new sweep.

Per arm (oracle_sub0, nonwinner_sub1), all at the LIVE re-reached
anchor observation (t2_s2110_d16) with ONE shared CRN noise tensor:
  full_lc        — ckpt action_out_proj + bias = LCProj(pool(history))
  lcproj_zero    — ckpt action_out_proj + bias = 0
  reset_state    — bias = LCProj(pool(initial_state at anchor only))
  shuffled_state — bias = LCProj(pool(SHA-seeded permuted history))
plus one stock reference (stock action_out_proj, no bias).

Env is stepped only to re-reach the anchor (deterministic stored-action
replay); no branch/evaluation rollout occurs, so no video is required —
generation is tensor-only. Output:
results/libero_loho_public_v1/<DATE>_v071_n1attr_r1/attribution.json
"""

from __future__ import annotations

import hashlib
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

from scripts.v071_0b_transmission_smoke import (ANCHOR, ARMS,  # noqa: E402
                                                LCProj, sha_seed)

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
SMOKE = RESULTS / "2026-08-02_v071_0b_smoke_r1"
STAGE, RUN_ID = "n1attr", "r1"
TASK = "loho_t2_basket3"
EPISODE_LEN = 900
SHUFFLE_SEED_PAYLOAD = "v071_n1attr_r1|shuffle_history"
NOISE_PAYLOAD = "v071_n1attr_r1|action"


def chunk_sha(chunk: torch.Tensor) -> str:
    return hashlib.sha256(
        chunk.float().cpu().numpy().tobytes()).hexdigest()[:16]


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import freeze_pi05_base, sample_chunks_lc
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.task_automaton import GoalAutomaton
    from lcwm.v067_lineage import flow_noise, sha256_file
    from lcwm.v06_model import V06State

    device = torch.device("cuda")
    date = subprocess.run(["date", "+%F"], capture_output=True,
                          text=True,
                          env={"TZ": "America/Chicago"}).stdout.strip()
    root = None
    for p in sorted(RESULTS.glob(f"*_v071_{STAGE}_{RUN_ID}")):
        root = p
    root = root or RESULTS / f"{date}_v071_{STAGE}_{RUN_ID}"
    root.mkdir(exist_ok=True)

    sid, d = ANCHOR
    source = torch.load(DATA / "corrections_sources_v069"
                        / f"{sid}.pt", weights_only=False)
    grp = torch.load(DATA / "corrections_v069" / f"{sid}_d{d}.pt",
                     weights_only=False)
    by_key = {str(b["key"]): b for b in grp["branch_summaries"]}
    targets = {arm: by_key[k]["chunk_norm"]
               for arm, k in ARMS.items()}
    u0_chunk = source["rows"][d]["chunk_norm"]

    model = V06State().to(device)
    wm = torch.load(RESULTS / "v069_predictive"
                    / "checkpoint_selected.pt", weights_only=False)
    model.load_state_dict(wm["model"])
    model.eval()
    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    freeze_pi05_base(runner.policy)
    stock_out_proj = {
        k: v.detach().clone() for k, v in
        runner.policy.model.action_out_proj.state_dict().items()}

    # ---- live re-reach to the smoke anchor (deterministic replay) ------
    env = make_public_env(TASK, EPISODE_LEN + 200)
    try:
        runner.reset()
        env.reset(seed=source["seed"])
        env._env.env.horizon = EPISODE_LEN + 300
        auto0 = GoalAutomaton(list(json.loads(
            (RESULTS / "goal_spec_manifest_v067.json").read_text())
            ["tasks"][TASK]["goal_specs"]["t2_canonical"]
            ["ordered_subgoals"]))
        auto0.start(env)
        auto0.evaluate(env, 0)
        t = 0
        for i, row in enumerate(source["rows"]):
            if i >= d:
                break
            for a_env in row["actions_env"]:
                env.step(a_env)
                t += 1
                auto0.evaluate(env, t)
        err = float(np.abs(body_positions(
            env, list(auto0.bodies.values()))
            - source["rows"][d]["obj_before"]).max())
        assert err < 2e-3, f"re-reach {err:.2e}"
        obs_live = env._format_raw_obs(
            env._env.env._get_observations())
    finally:
        env.close()

    batch_live = runner._obs_to_policy_batch(
        obs_live, source["language_canonical"])
    prefix_live = prefix_forward(runner.policy, batch_live)

    # ---- recurrent pools: full / reset / shuffled ----------------------
    def pool_over(order: list[int]) -> torch.Tensor:
        """Recurrent z over stored history rows in the given order,
        finishing at the LIVE anchor prefix."""
        z = None
        prev_row = None
        for i in order:
            row = source["rows"][i]
            b = runner._obs_to_policy_batch(
                row["obs"], source["language_canonical"])
            pfx = prefix_forward(runner.policy, b)
            h, m = pfx.hidden.float(), pfx.pad_masks.bool()
            if z is None:
                z = model.initial_state(h, m)
            else:
                aa = prev_row["chunk_norm"][None, :10].float() \
                    .to(device)
                am = (torch.arange(10, device=device)[None]
                      < prev_row["executed_len"])
                z = model.step(z, aa, h, m, action_mask=am)
            prev_row = row
        h, m = prefix_live.hidden.float(), prefix_live.pad_masks.bool()
        if z is None:
            z = model.initial_state(h, m)
        else:
            aa = prev_row["chunk_norm"][None, :10].float().to(device)
            am = (torch.arange(10, device=device)[None]
                  < prev_row["executed_len"])
            z = model.step(z, aa, h, m, action_mask=am)
        return z.mean(dim=1).detach()

    g = torch.Generator().manual_seed(sha_seed(SHUFFLE_SEED_PAYLOAD))
    perm = torch.randperm(d, generator=g).tolist()
    pools = {"full": pool_over(list(range(d))),
             "reset": pool_over([]),
             "shuffled": pool_over(perm)}

    noise = flow_noise(sha_seed(NOISE_PAYLOAD), cfg.chunk_size,
                       cfg.max_action_dim).to(device)

    def generate(bias) -> torch.Tensor:
        return sample_chunks_lc(runner.policy, batch_live, bias, n=1,
                                noise=noise,
                                prefix=prefix_live)[0].float().cpu()

    results = {"conditions": {}, "pairwise_l1_first10": {}}
    chunks = {}
    runner.policy.model.action_out_proj.load_state_dict(
        stock_out_proj)
    chunks["stock"] = sample_chunks(
        runner.policy, batch_live, n=1, noise=noise,
        prefix=prefix_live)[0].float().cpu()

    for arm in ARMS:
        ckpt = torch.load(SMOKE / "checkpoints" / f"{arm}_final.pt",
                          weights_only=False)
        lc_proj = LCProj().to(device)
        lc_proj.load_state_dict(ckpt["lc_proj"])
        runner.policy.model.action_out_proj.load_state_dict(
            ckpt["action_out_proj"])
        conds = {
            "full_lc": lc_proj(pools["full"]),
            "lcproj_zero": torch.zeros(1, 1024, device=device),
            "reset_state": lc_proj(pools["reset"]),
            "shuffled_state": lc_proj(pools["shuffled"]),
        }
        for cname, bias in conds.items():
            chunks[f"{arm}/{cname}"] = generate(bias)
    runner.policy.model.action_out_proj.load_state_dict(
        stock_out_proj)

    refs = {"target_oracle": targets["oracle_sub0"].float(),
            "target_nonwinner": targets["nonwinner_sub1"].float(),
            "u0_stored": u0_chunk.float()}
    for key, ch in chunks.items():
        row = {"chunk_sha256_16": chunk_sha(ch)}
        for rname, ref in refs.items():
            row[f"l1_first10_vs_{rname}"] = float(
                (ch[:10] - ref[:10]).abs().mean())
        results["conditions"][key] = row
    keys = sorted(chunks)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            results["pairwise_l1_first10"][f"{a} vs {b}"] = float(
                (chunks[a][:10] - chunks[b][:10]).abs().mean())

    out = {
        "schema": "v071_n1attr_v1", "run_schema": "v071",
        "run_id": RUN_ID, "stage": STAGE, "run_date": date,
        "anchor": f"{sid}_d{d}",
        "note": ("bounded attribution measurement on existing smoke "
                 "checkpoints; no training, no selection; env stepped "
                 "only for deterministic re-reach — no evaluation "
                 "rollout, tensor-only generation, no video required"),
        "crn": {"noise_payload": NOISE_PAYLOAD,
                "shuffle_payload": SHUFFLE_SEED_PAYLOAD,
                "history_permutation": perm},
        "checkpoints": {arm: sha256_file(
            SMOKE / "checkpoints" / f"{arm}_final.pt")
            for arm in ARMS},
        "predictive_sha256": sha256_file(
            RESULTS / "v069_predictive" / "checkpoint_selected.pt"),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=REPO_ROOT).stdout.strip(),
        **results,
    }
    (root / "attribution.json").write_text(json.dumps(out, indent=2))
    torch.save({k: v for k, v in chunks.items()},
               root / "generated_chunks.pt")
    for k in keys:
        print(k, results["conditions"][k]["chunk_sha256_16"],
              flush=True)
    print(f"-> {root}", flush=True)


if __name__ == "__main__":
    main()
