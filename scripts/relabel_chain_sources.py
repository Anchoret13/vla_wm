#!/usr/bin/env python
"""P3 relabel pass (registered 2026-07-25.md): deterministic replay of the
chain source episodes, capturing per-decision labels aligned to the stored
histories.

The chain rollouts are exactly reproducible (same env seed + per-decision
flow noise `base + decision`), so this adds q/obj_pos/predicate-bit labels
and full-episode continuation (for the discounted atom-completion V^{pi0}
target, generating policy = frozen pi0.5, full prompt) to episodes that are
already training sources — no new interaction states. Sidecar artifacts:
datasets/chain_source_labels_v1/<source_id>.pt
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

OUT_DIR = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/chain_source_labels_v1"
)
BRANCH_DIR = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/chain_branches_v1_1"
)
SOURCE_RE = re.compile(r"chain3_lr2-full-seed(\d+)-src(\d+)")


@torch.no_grad()
def relabel(runner, env, seed: int, noise_base: int) -> dict:
    from lcwm.branch_collect import _execute_env_block
    from lcwm.probe_data import body_positions, discover_object_bodies
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.seq_data import goal_atoms, predicate_bits

    runner.reset()
    obs, _ = env.reset(seed=seed)
    atoms = goal_atoms(env)
    bodies = discover_object_bodies(env)
    body_names = list(bodies.values())
    records = []

    def capture(t, decision, terminal):
        records.append(
            {
                "t": t,
                "decision": decision,
                "q": torch.from_numpy(
                    np.concatenate(
                        [
                            obs["robot_state"]["eef"]["pos"],
                            obs["robot_state"]["eef"]["quat"],
                            obs["robot_state"]["gripper"]["qpos"],
                        ]
                    )
                ).float(),
                "obj_pos": torch.from_numpy(
                    body_positions(env, body_names)
                ).float(),
                "bits": torch.from_numpy(predicate_bits(env, atoms)),
                "terminal": terminal,
            }
        )

    t = decision = 0
    terminated = truncated = False
    while t < env.episode_length:
        capture(t, decision, terminal=False)
        batch = runner._obs_to_policy_batch(obs, env.task_description)
        prefix = prefix_forward(runner.policy, batch)
        chunk = sample_chunks(
            runner.policy, batch, n=1,
            seed=noise_base + decision, prefix=prefix,
        )
        actions_env = runner.chunk_to_env(chunk[:, :10])
        obs, executed, terminated, truncated, _r, _i = _execute_env_block(
            env, actions_env, 10
        )
        t += executed
        decision += 1
        if terminated or truncated:
            break
    capture(t, decision, terminal=True)
    return {
        "schema": "chain_source_labels_v1_20260725",
        "task": "chain3_lr2",
        "seed": seed,
        "policy_noise_base": noise_base,
        "generating_policy": "pi05_full_prompt_frozen",
        "language": env.task_description,
        "goal_atoms": atoms,
        "object_names": list(bodies),
        "t": torch.tensor([r["t"] for r in records]),
        "decision": torch.tensor([r["decision"] for r in records]),
        "q": torch.stack([r["q"] for r in records]),
        "obj_pos": torch.stack([r["obj_pos"] for r in records]),
        "bits": torch.stack([r["bits"] for r in records]),
        "terminal": torch.tensor([r["terminal"] for r in records]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.loho import make_chain_env

    manifest = json.loads(
        (BRANCH_DIR / "branch_manifest.json").read_text()
    )
    sources = sorted(manifest["source_episode_splits"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = Pi05Runner(suite_name="libero_10")
    for source_id in sources:
        match = SOURCE_RE.fullmatch(source_id)
        if match is None:
            print(f"[skip] unparseable source id {source_id}", flush=True)
            continue
        out = OUT_DIR / f"{source_id}.pt"
        if out.exists():
            print(f"[skip] exists {out.name}", flush=True)
            continue
        seed, noise = int(match.group(1)), int(match.group(2))
        env = make_chain_env("chain3_lr2")
        try:
            record = relabel(runner, env, seed, noise)
        finally:
            env.close()
        record["source_trajectory_id"] = source_id
        record["split"] = manifest["source_episode_splits"][source_id]
        torch.save(record, out)
        bits_gained = int(record["bits"][-1].sum())
        print(
            f"[relabel] {source_id}: {record['q'].shape[0]} records, "
            f"final bits sum={bits_gained} -> {out.name}",
            flush=True,
        )
    print("relabel complete", flush=True)


if __name__ == "__main__":
    main()
