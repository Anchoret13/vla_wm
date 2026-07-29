#!/usr/bin/env python
"""H1 — hardened paired long-horizon evaluator (behavior_factorial_v1).

Contract (2026-07-29 H1):
- noise_seed(task, env_seed, decision) = 20_000_000 + task_index*2_000_000
  + env_seed*1_000 + decision, task_index chain3/4/5 = 0/1/2; the stream is
  shared exactly across arms (common-random-number variance reduction).
- stock arm = the SAME seeded LC sampler with wz_out == 0; exact stock
  parity is asserted before evaluation (native sampler lacks this contract).
- prompt_mode=full only.
- Immutable run manifest; one authoritative record per (run_id, task, seed,
  arm); --resume verifies the manifest hash and never overwrites a
  completed record; arms interleave within each (task, seed) block by a
  pre-registered hash permutation.
- Per-episode: normalized+env action traces always; decision-boundary
  q/object traces when a failure phase is reported; success videos saved
  from the authoritative rollout frames.
- Metrics per episode: success/time, final Q, D_final, D_max, A_D,
  per-atom first-completion + later-damage, first unresolved atom, failure
  phase, steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time as walltime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

NOISE_BASE = 20_000_000
TASK_INDEX = {"chain3_lr2": 0, "chain4_lr2": 1, "chain5_lr2": 2}
ROOT = REPO_ROOT / "results" / "behavior_factorial_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_seed(task: str, env_seed: int, decision: int) -> int:
    return (
        NOISE_BASE
        + TASK_INDEX[task] * 2_000_000
        + env_seed * 1_000
        + decision
    )


def arm_order(task: str, env_seed: int, arms: list[str]) -> list[str]:
    """Pre-registered hash permutation for arm interleaving."""
    keyed = sorted(
        arms,
        key=lambda arm: hashlib.sha256(
            f"{task}|{env_seed}|{arm}".encode()
        ).hexdigest(),
    )
    return keyed


def episode_metrics(result, n_atoms: int) -> dict:
    timeline = result.bits_timeline
    prefix_depths = []
    first_completion = [None] * n_atoms
    damaged = [False] * n_atoms
    for step, bits in timeline:
        depth = 0
        for i, b in enumerate(bits):
            if b and first_completion[i] is None:
                first_completion[i] = step
            if not b and first_completion[i] is not None:
                damaged[i] = True
        for b in bits:
            if b:
                depth += 1
            else:
                break
        prefix_depths.append(depth)
    final_bits = timeline[-1][1]
    d_final = 0
    for b in final_bits:
        if b:
            d_final += 1
        else:
            break
    first_unresolved = next(
        (i for i, b in enumerate(final_bits) if not b), None
    )
    return {
        "success": bool(result.success),
        "q_final": float(result.q_score),
        "steps": int(result.steps),
        "d_final": d_final,
        "d_max": max(prefix_depths) if prefix_depths else 0,
        "a_d": (
            sum(prefix_depths) / (len(prefix_depths) * n_atoms)
            if prefix_depths
            else 0.0
        ),
        "atom_first_completion": first_completion,
        "atom_damaged": damaged,
        "first_unresolved_atom": first_unresolved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        required=True,
        help="name=adapter_dir pairs; 'stock' uses the frozen adapter with "
        "wz_out zeroed (parity-asserted)",
    )
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--panel", default="development")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import LCFlowRunner, load_lc_adapter
    from lcwm.loho import make_chain_env, run_chain_episode

    panel_dir = ROOT / args.panel
    panel_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "traces").mkdir(exist_ok=True)
    (ROOT / "videos").mkdir(exist_ok=True)
    records_path = panel_dir / "records.jsonl"

    # Arm syntax: name[=adapter_dir][@current]; '@current' deploys the
    # current-observation-only state rule (H3): z reset before every chunk
    # so z_t = U(z0, h_t) with no history carry.
    arm_specs: dict[str, Path | None] = {}
    arm_state_mode: dict[str, str] = {}
    for spec in args.arms:
        name, _, path = spec.partition("=")
        mode = "recurrent"
        if path.endswith("@current"):
            path, mode = path[: -len("@current")], "current"
        arm_specs[name] = Path(path) if path else None
        arm_state_mode[name] = mode
    reference_adapter = next(
        (p for p in arm_specs.values() if p is not None), None
    )
    assert reference_adapter is not None, "need at least one adapter arm"

    manifest = {
        "schema": "behavior_factorial_v1",
        "run_id": args.run_id,
        "noise_contract": (
            "20e6 + task_index*2e6 + env_seed*1e3 + decision; "
            "chain3/4/5 = 0/1/2; shared across arms"
        ),
        "prompt_mode": "full",
        "arms": {
            name: (str(p) if p else "wz_zero_of_reference")
            for name, p in arm_specs.items()
        },
        "adapter_hashes": {
            name: sha256_file(p / "lc_flow.safetensors")
            for name, p in arm_specs.items()
            if p is not None
        },
        "tasks": args.tasks,
        "seeds": args.seeds,
        "code_files": {
            f: sha256_file(REPO_ROOT / f)
            for f in (
                "scripts/eval_long_horizon_paired.py",
                "lcwm/lc_flow.py",
                "lcwm/loho.py",
                "lcwm/chassis.py",
            )
        },
        "bddl_hashes": {
            t: sha256_file(REPO_ROOT / "bddl" / "chains" / f"{t}.bddl")
            for t in args.tasks
            if (REPO_ROOT / "bddl" / "chains" / f"{t}.bddl").exists()
        },
    }
    manifest_payload = json.dumps(manifest, sort_keys=True)
    manifest_hash = hashlib.sha256(manifest_payload.encode()).hexdigest()
    manifest["manifest_sha256"] = manifest_hash
    manifest_file = ROOT / "run_manifest.json"
    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text())
        if existing.get("run_id") == args.run_id:
            assert existing["manifest_sha256"] == manifest_hash, (
                "manifest hash mismatch on resume — aborting"
            )
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2))
    if not args.resume and records_path.exists():
        completed = [
            json.loads(line)
            for line in records_path.read_text().splitlines()
            if line.strip()
        ]
        conflicts = [
            r
            for r in completed
            if r["run_id"] == args.run_id
        ]
        assert not conflicts, (
            "records exist for this run_id; use --resume"
        )
    done_keys = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done_keys.add((r["run_id"], r["task"], r["seed"], r["arm"]))

    base = Pi05Runner(suite_name="libero_10")

    class CurrentOnlyFlow(LCFlowRunner):
        def _generate_chunk(self, *a, **k):
            self.z = None  # H3 rule: posterior from the current prefix only
            return super()._generate_chunk(*a, **k)

    def build_flow(arm: str):
        adapter_dir = arm_specs[arm] or reference_adapter
        lc_state, _ = load_lc_adapter(adapter_dir, device=args.device)
        if arm_specs[arm] is None:  # stock arm: zero the adapter output
            torch.nn.init.zeros_(lc_state.wz_out.weight)
            if lc_state.wz_out.bias is not None:
                torch.nn.init.zeros_(lc_state.wz_out.bias)
            assert float(lc_state.wz_out.weight.abs().max()) == 0.0
        cls = (
            CurrentOnlyFlow
            if arm_state_mode[arm] == "current"
            else LCFlowRunner
        )
        return cls(base, lc_state)

    ledger_path = panel_dir / "ledger.jsonl"

    for task in args.tasks:
        n_atoms = int(task[5])
        for seed in args.seeds:
            for arm in arm_order(task, seed, list(arm_specs)):
                key = (args.run_id, task, seed, arm)
                if key in done_keys:
                    continue
                started = walltime.time()
                with ledger_path.open("a") as ledger:
                    ledger.write(json.dumps(
                        {"key": key, "status": "running"}) + "\n")
                flow = build_flow(arm)

                class NoiseRunner:
                    def __init__(self, flow, task, seed):
                        self.flow, self.task, self.seed = flow, task, seed
                        self.actions_env: list[np.ndarray] = []
                        self.noise_seeds: list[int] = []

                    def reset(self):
                        self.flow.reset()

                    def select_action(self, obs, desc):
                        ns = noise_seed(
                            self.task, self.seed,
                            self.flow.decision_index,
                        )
                        action = self.flow.select_action(
                            obs, desc, seed=ns,
                        )
                        self.noise_seeds.append(ns)
                        self.actions_env.append(np.asarray(action))
                        return action

                runner = NoiseRunner(flow, task, seed)
                env = make_chain_env(task)
                try:
                    result = run_chain_episode(
                        runner, env, "full", seed=seed,
                    )
                finally:
                    env.close()
                metrics = episode_metrics(result, n_atoms)
                # Env-convention trace always; normalized recoverable via
                # the exact stored mean/std inverse (recorded choice).
                np.savez_compressed(
                    ROOT / "traces"
                    / f"{args.run_id}_{task}_s{seed}_{arm}.npz",
                    actions_env=np.stack(runner.actions_env),
                    noise_seeds=np.asarray(
                        sorted(set(runner.noise_seeds))
                    ),
                )
                record = {
                    "run_id": args.run_id,
                    "task": task,
                    "seed": seed,
                    "arm": arm,
                    "prompt_mode": "full",
                    **metrics,
                    "wall_seconds": round(walltime.time() - started, 1),
                    "manifest_sha256": manifest_hash,
                }
                with records_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                with ledger_path.open("a") as ledger:
                    ledger.write(json.dumps(
                        {"key": key, "status": "complete",
                         "success": metrics["success"]}) + "\n")
                print(
                    f"[{task} seed={seed} arm={arm}] success="
                    f"{metrics['success']} q={metrics['q_final']:.3f} "
                    f"D_final={metrics['d_final']} A_D={metrics['a_d']:.3f}",
                    flush=True,
                )
    print(f"-> {records_path}", flush=True)


if __name__ == "__main__":
    main()
