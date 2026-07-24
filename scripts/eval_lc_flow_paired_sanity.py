#!/usr/bin/env python
"""One-seed paired closed-loop sanity for a trained LC-Flow adapter.

This is an integration/rejection gate, not a performance estimate.  The
zero-initialized LC control and trained adapter use the same frozen PI05 base,
environment seed, full composite instruction, and decision-indexed flow-noise
schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.chassis import Pi05Runner  # noqa: E402
from lcwm.lc_flow import (  # noqa: E402
    LCFlowRunner,
    LCState,
    load_lc_adapter,
)
from lcwm.loho import make_chain_env, run_chain_episode  # noqa: E402


class DecisionSeededLCFlow:
    """Supply one reproducible PI05 flow-noise seed per generated chunk."""

    def __init__(self, flow: LCFlowRunner, noise_seed_base: int):
        self.flow = flow
        self.noise_seed_base = int(noise_seed_base)
        self.noise_seeds: list[int] = []

    @property
    def decision_index(self) -> int:
        return self.flow.decision_index

    def reset(self) -> None:
        self.flow.reset()
        self.noise_seeds.clear()

    def select_action(self, obs: dict, task_description: str):
        before = self.flow.decision_index
        seed = self.noise_seed_base + before
        action = self.flow.select_action(
            obs,
            task_description,
            seed=seed,
        )
        if self.flow.decision_index != before:
            self.noise_seeds.append(seed)
        return action


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--task", default="chain3_lr2")
    parser.add_argument("--seed", type=int, default=1028)
    parser.add_argument("--noise-seed-base", type=int, default=9_028_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trained, metadata = load_lc_adapter(args.adapter, device="cpu")
    base_model_id = metadata.get(
        "base_model_id", "lerobot/pi05_libero_finetuned"
    )
    zero = LCState(trained.config)
    if int(torch.count_nonzero(zero.wz_out.weight)) != 0 or int(
        torch.count_nonzero(zero.wz_out.bias)
    ) != 0:
        raise AssertionError("fresh LC control must have an exactly zero W_z")

    base = Pi05Runner(
        model_id=base_model_id,
        suite_name="libero_10",
        device=args.device,
    )
    records: dict[str, dict] = {}
    schedules: dict[str, list[int]] = {}
    treatments = (
        ("zero_lc_custom_fixed_noise", zero),
        ("trained_lc", trained),
    )
    for name, state in treatments:
        seeded = DecisionSeededLCFlow(
            LCFlowRunner(base, state),
            args.noise_seed_base,
        )
        env = make_chain_env(args.task)
        try:
            result = run_chain_episode(
                seeded,
                env,
                condition="full",
                seed=args.seed,
            )
        finally:
            env.close()
        records[name] = asdict(result)
        schedules[name] = seeded.noise_seeds
        print(
            f"[{name}] success={result.success} q={result.q_score:.3f} "
            f"steps={result.steps} decisions={len(seeded.noise_seeds)}",
            flush=True,
        )

    baseline_schedule = schedules["zero_lc_custom_fixed_noise"]
    trained_schedule = schedules["trained_lc"]
    paired_prefix = min(len(baseline_schedule), len(trained_schedule))
    common_noise_prefix_equal = (
        baseline_schedule[:paired_prefix]
        == trained_schedule[:paired_prefix]
    )
    if not common_noise_prefix_equal:
        raise AssertionError("decision-indexed flow-noise schedules diverged")

    baseline = records["zero_lc_custom_fixed_noise"]
    treatment = records["trained_lc"]
    output = {
        "schema": "lc_flow_paired_closed_loop_sanity_v1",
        "claim_boundary": (
            "single unseen dev seed; integration/rejection gate only, "
            "not a policy-performance estimate"
        ),
        "task": args.task,
        "condition": "full",
        "environment_seed": args.seed,
        "treatment_order": [name for name, _ in treatments],
        "flow_noise_contract": {
            "formula": "noise_seed_base + logical_decision_index",
            "noise_seed_base": args.noise_seed_base,
            "common_prefix_length": paired_prefix,
            "common_noise_prefix_equal": common_noise_prefix_equal,
            "schedules": schedules,
        },
        "adapter": {
            "directory": str(args.adapter.resolve()),
            "weights_sha256": sha256(args.adapter / "lc_flow.safetensors"),
            "config_sha256": sha256(args.adapter / "lc_flow_config.json"),
            "metadata": metadata,
        },
        "results": records,
        "delta_trained_minus_zero": {
            "success": int(treatment["success"]) - int(baseline["success"]),
            "q_score": treatment["q_score"] - baseline["q_score"],
            "steps": treatment["steps"] - baseline["steps"],
        },
        "known_limits": [
            "one held-out initialization seed",
            "same Chain-3 instruction as training",
            "five positive branch-FM targets from mixed collection contracts",
            "three train groups share source episode seed 1001",
            "no demonstration rehearsal or LIBERO-10 retention check",
            "common noise after trajectory divergence is variance reduction, "
            "not a matched-state intervention",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"-> {args.output}", flush=True)


if __name__ == "__main__":
    main()
