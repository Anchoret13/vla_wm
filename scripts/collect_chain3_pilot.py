#!/usr/bin/env python
"""Collect the first grouped Chain-3 stall/recovery branch pilot.

The script intentionally does not start a large collection by default.  It
collects one group (seed 1000) unless explicit seeds are supplied.  Each group
contains one stock reference plus four independently sampled proposals from
each of the full, exact-pair, and remaining-atomic prompts. All thirteen
branches are restored to the same decision-boundary snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHAIN3_PAIR_INSTRUCTION = (
    "put both the alphabet soup and the tomato sauce in the basket"
)
CHAIN3_TARGET_SUPPORT_INSTRUCTION = (
    "put both the cream cheese box and the butter in the basket"
)
COLLECTION_CONTRACT_VERSION = "chain3_branch_v1_1_20260723"
COLLECTOR_SOURCE_FILES = (
    "scripts/collect_chain3_pilot.py",
    "lcwm/branch_collect.py",
    "lcwm/branch_data.py",
    "lcwm/snapshot.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get(
                "LCWM_BRANCH_DIR",
                REPO_ROOT.parent / "datasets" / "chain_branches_v1_1",
            )
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000])
    parser.add_argument(
        "--full-count",
        type=int,
        default=4,
        help="new full-prompt proposals, excluding the stock reference",
    )
    parser.add_argument("--pair-count", type=int, default=4)
    parser.add_argument("--remaining-count", type=int, default=4)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument(
        "--recovery-warmup-blocks",
        type=int,
        default=1,
        help=(
            "maximum oracle-mined remaining-goal blocks before branching; "
            "0 reproduces the weak post-anchor snapshot"
        ),
    )
    parser.add_argument("--recovery-candidates", type=int, default=4)
    parser.add_argument(
        "--recovery-anchor-blocks",
        type=int,
        default=24,
        help=(
            "compatible cream-cheese+butter pair-prompt bridge blocks used "
            "before target-prompt mining"
        ),
    )
    parser.add_argument(
        "--recovery-anchor-stop-distance",
        type=float,
        default=0.16,
    )
    parser.add_argument(
        "--recovery-anchor-admission-distance",
        type=float,
        default=0.25,
        help=(
            "maximum safe anchor distance allowed to enter the final "
            "remaining-goal recovery stage"
        ),
    )
    parser.add_argument(
        "--recovery-stop-distance",
        type=float,
        default=0.10,
        help="stop recovery mining when EEF-target distance reaches this many m",
    )
    parser.add_argument("--continuation-horizon", type=int, default=160)
    parser.add_argument(
        "--continuation-count",
        type=int,
        default=4,
        help=(
            "number of branches receiving an atomic continuation label; "
            "0 disables this expensive target"
        ),
    )
    parser.add_argument(
        "--source-attempts",
        type=int,
        default=4,
        help=(
            "full-prompt source rollouts tried per environment seed before "
            "declaring that the 2/3 branch state is unavailable"
        ),
    )
    parser.add_argument(
        "--source-attempt-stride",
        type=int,
        default=100,
        help="noise-seed stride between full-prompt source attempts",
    )
    parser.add_argument("--source-noise-seed", type=int, default=20_000)
    parser.add_argument("--proposal-noise-seed", type=int, default=30_000)
    parser.add_argument("--continuation-noise-seed", type=int, default=40_000)
    parser.add_argument("--recovery-noise-seed", type=int, default=50_000)
    parser.add_argument(
        "--recovery-anchor-noise-seed", type=int, default=60_000
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing group for the same source seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved collection/split contract without loading PI05",
    )
    return parser.parse_args()


def assign_splits(seeds: list[int]) -> dict[int, str]:
    """Stable source-episode split, independent of call order and reruns."""
    unique = list(dict.fromkeys(seeds))
    return {
        seed: (
            "dev"
            if seed % 10 == 8
            else "test"
            if seed % 10 == 9
            else "train"
        )
        for seed in unique
    }


def noise_base_for_seed(base: int, seed: int, *, origin: int = 1_000) -> int:
    """Resolve a stable per-source noise block independent of CLI order.

    Seed 1000 intentionally retains the already-audited base-noise rollout.
    """
    value = base + (seed - origin) * 1_000
    if value < 0:
        raise ValueError("resolved noise seed must be nonnegative")
    return value


def collector_source_sha256() -> str:
    """Hash the exact uncommitted collector sources used by an artifact."""
    digest = hashlib.sha256()
    for relative in COLLECTOR_SOURCE_FILES:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((REPO_ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def seed_collection_contract(
    args: argparse.Namespace,
    seed: int,
    *,
    base_model_id: str,
    num_inference_steps: int,
) -> dict:
    """Invocation-order-independent contract for one source episode."""
    return {
        "version": COLLECTION_CONTRACT_VERSION,
        "collector_source_sha256": collector_source_sha256(),
        "task": "chain3_lr2",
        "source_seed": seed,
        "split": assign_splits([seed])[seed],
        "base_model_id": base_model_id,
        "num_inference_steps": num_inference_steps,
        "proposal_counts": {
            "stock_full": 1,
            "full": args.full_count,
            "exact_pair": args.pair_count,
            "remaining_atomic": args.remaining_count,
            "optional_mined_successor": 1,
        },
        "proposal_instructions": {
            "exact_pair": CHAIN3_PAIR_INSTRUCTION,
            "support_pair_recovery": CHAIN3_TARGET_SUPPORT_INSTRUCTION,
        },
        "execution_horizon": args.execution_horizon,
        "source_search": {
            "attempts": args.source_attempts,
            "attempt_stride": args.source_attempt_stride,
        },
        "recovery": {
            "warmup_blocks": args.recovery_warmup_blocks,
            "anchor_blocks": args.recovery_anchor_blocks,
            "candidates_per_block": args.recovery_candidates,
            "stop_distance": args.recovery_stop_distance,
            "anchor_stop_distance": args.recovery_anchor_stop_distance,
            "anchor_admission_distance": (
                args.recovery_anchor_admission_distance
            ),
        },
        "continuation": {
            "horizon": args.continuation_horizon,
            "count": args.continuation_count,
        },
        "noise_seeds": {
            "source": noise_base_for_seed(args.source_noise_seed, seed),
            "proposal": noise_base_for_seed(args.proposal_noise_seed, seed),
            "continuation": noise_base_for_seed(
                args.continuation_noise_seed, seed
            ),
            "recovery": noise_base_for_seed(args.recovery_noise_seed, seed),
            "recovery_anchor": noise_base_for_seed(
                args.recovery_anchor_noise_seed, seed
            ),
        },
    }


def contract_sha256(contract: dict) -> str:
    payload = json.dumps(
        contract, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def branch_coverage_summary(group: dict) -> dict[str, int]:
    """Compact, JSON-safe signal coverage for live collection decisions."""
    import torch

    object_displacement = torch.linalg.vector_norm(
        group["object_pos_after"] - group["object_pos_before"][None],
        dim=-1,
    ).amax(dim=1)
    active_atoms = group["atom_mask"].bool()
    predicate_flip = (
        group["bits_after"][:, active_atoms]
        != group["bits_before"][active_atoms][None]
    ).any(dim=1)
    margin = float(group["dense_progress_margin"])
    return {
        "branches": int(group["actions_norm"].shape[0]),
        "object_effectful_branches": int(
            (object_displacement > 5e-4).sum()
        ),
        "predicate_flip_branches": int(predicate_flip.sum()),
        "absolute_dense_progress_branches": int(
            (group["dense_goal_progress"] > margin).sum()
        ),
        "relative_superior_branches": int(
            group["immediate_beats_stock"].sum()
        ),
        "positive_flow_targets": int(
            (group["branch_weights"] > 0).sum()
        ),
    }


def resolved_contract(args: argparse.Namespace) -> dict:
    if min(args.full_count, args.pair_count, args.remaining_count) < 0:
        raise ValueError("proposal counts must be nonnegative")
    if args.execution_horizon != 10:
        raise ValueError("LC-Flow v0 uses an exact 10-action commitment")
    if args.recovery_warmup_blocks < 0:
        raise ValueError("--recovery-warmup-blocks must be nonnegative")
    if args.recovery_candidates < 1:
        raise ValueError("--recovery-candidates must be positive")
    if args.recovery_anchor_blocks < 0:
        raise ValueError("--recovery-anchor-blocks must be nonnegative")
    if args.recovery_stop_distance <= 0:
        raise ValueError("--recovery-stop-distance must be positive")
    if args.recovery_anchor_stop_distance <= 0:
        raise ValueError("--recovery-anchor-stop-distance must be positive")
    if (
        args.recovery_anchor_admission_distance
        < args.recovery_anchor_stop_distance
    ):
        raise ValueError(
            "--recovery-anchor-admission-distance must be at least the "
            "anchor stop distance"
        )
    if args.continuation_horizon < 1:
        raise ValueError("--continuation-horizon must be positive")
    if not 1 <= args.source_attempts <= 10:
        raise ValueError("--source-attempts must be in [1,10]")
    if args.source_attempt_stride < 100:
        raise ValueError("--source-attempt-stride must be at least 100")
    if (args.source_attempts - 1) * args.source_attempt_stride >= 1_000:
        raise ValueError("source attempts would cross the next seed's noise block")
    branch_count = (
        1 + args.full_count + args.pair_count + args.remaining_count
    )
    if not 0 <= args.continuation_count <= branch_count:
        raise ValueError("--continuation-count must be within the branch pool")
    split_by_seed = assign_splits(args.seeds)
    return {
        "contract_version": COLLECTION_CONTRACT_VERSION,
        "collector_source_sha256": collector_source_sha256(),
        "task": "chain3_lr2",
        "output": str(args.output.resolve()),
        "source_seeds": list(dict.fromkeys(args.seeds)),
        "split_by_seed": {
            str(seed): split_by_seed[seed] for seed in split_by_seed
        },
        "base_branches_per_group": branch_count,
        "maximum_branches_per_group": branch_count + 1,
        "proposal_counts": {
            "stock_full": 1,
            "full": args.full_count,
            "exact_pair": args.pair_count,
            "remaining_atomic": args.remaining_count,
            "optional_mined_successor": 1,
        },
        "execution_horizon": args.execution_horizon,
        "source_attempts": args.source_attempts,
        "source_attempt_stride": args.source_attempt_stride,
        "recovery_warmup_blocks": args.recovery_warmup_blocks,
        "recovery_candidates_per_block": args.recovery_candidates,
        "recovery_stop_distance": args.recovery_stop_distance,
        "recovery_anchor": {
            "goal": "libero10_scene2_cream_cheese_butter_pair",
            "max_blocks": args.recovery_anchor_blocks,
            "stop_target_distance": args.recovery_anchor_stop_distance,
            "admission_distance": args.recovery_anchor_admission_distance,
            "role": "off-policy data-collection bridge; provenance only",
        },
        "continuation_horizon": args.continuation_horizon,
        "continuation_count": args.continuation_count,
        "split_rule": "seed_mod_10: 0..7=train, 8=dev, 9=test",
        "noise_rule": "base + (source_seed - 1000) * 1000",
        "noise_seed_bases": {
            "source": args.source_noise_seed,
            "proposal": args.proposal_noise_seed,
            "continuation": args.continuation_noise_seed,
            "recovery": args.recovery_noise_seed,
            "recovery_anchor": args.recovery_anchor_noise_seed,
        },
    }


def write_manifest(
    output: Path,
    contract: dict,
    paths_by_split: dict[str, list[str]],
    path_metadata: dict[str, dict],
    collection_attempts: dict[str, dict],
) -> None:
    from lcwm.branch_data import MANIFEST, SCHEMA

    target = output / MANIFEST
    existing: dict = {}
    if target.exists():
        existing = json.loads(target.read_text())
        if existing.get("schema") != SCHEMA:
            raise ValueError(
                f"refusing to merge manifest schema {existing.get('schema')!r}"
            )

    current_paths = {
        path for paths in paths_by_split.values() for path in paths
    }
    merged_splits = {}
    for split in ("train", "dev", "test"):
        retained = set(existing.get("splits", {}).get(split, []))
        retained.difference_update(current_paths)
        retained.update(paths_by_split[split])
        merged_splits[split] = sorted(retained)

    contracts = list(existing.get("collection_contracts", []))
    legacy_contract = existing.get("contract")
    if legacy_contract and legacy_contract not in contracts:
        contracts.append(legacy_contract)
    if contract not in contracts:
        contracts.append(contract)
    merged_metadata = dict(existing.get("path_metadata", {}))
    merged_metadata.update(path_metadata)
    merged_attempts = dict(existing.get("collection_attempts", {}))
    merged_attempts.update(collection_attempts)
    source_episode_splits = dict(existing.get("source_episode_splits", {}))
    for path, metadata in path_metadata.items():
        source = metadata["source_trajectory_id"]
        split = metadata["split"]
        previous = source_episode_splits.setdefault(source, split)
        if previous != split:
            raise ValueError(
                f"source episode {source!r} would cross {previous}/{split}"
            )
    manifest = {
        "schema": SCHEMA,
        "task": contract["task"],
        "grouping": "one item per snapshot; all sibling branches share a split",
        "collection_contracts": contracts,
        "splits": merged_splits,
        "source_episode_splits": source_episode_splits,
        "path_metadata": merged_metadata,
        "collection_attempts": merged_attempts,
    }
    temporary = output / f".{MANIFEST}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(target)


def main() -> None:
    args = parse_args()
    contract = resolved_contract(args)
    print(json.dumps(contract, indent=2), flush=True)
    if args.dry_run:
        return

    # Delay heavyweight imports so --dry-run remains a CPU/config-only check.
    from lcwm.branch_collect import (
        ProposalSpec,
        advance_recovery_context,
        collect_snapshot_group,
        rollout_to_stall,
    )
    from lcwm.branch_data import load_branch_group, save_branch_group
    from lcwm.chassis import Pi05Runner
    from lcwm.loho import SUBGOAL_PHRASE, make_chain_env
    from lcwm.probe_data import discover_object_bodies
    from lcwm.seq_data import goal_atoms, predicate_bits

    output = args.output.resolve()
    groups_dir = output / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    split_by_seed = assign_splits(args.seeds)
    paths_by_split = {"train": [], "dev": [], "test": []}
    path_metadata: dict[str, dict] = {}
    collection_attempts: dict[str, dict] = {}
    runner = Pi05Runner(suite_name="libero_10")
    model_key = hashlib.sha1(runner.model_id.encode()).hexdigest()[:8]

    for seed in dict.fromkeys(args.seeds):
        seed_contract = seed_collection_contract(
            args,
            seed,
            base_model_id=runner.model_id,
            num_inference_steps=runner.policy.config.num_inference_steps,
        )
        contract_hash = contract_sha256(seed_contract)
        source_noise = noise_base_for_seed(args.source_noise_seed, seed)
        proposal_noise = noise_base_for_seed(args.proposal_noise_seed, seed)
        continuation_noise = noise_base_for_seed(
            args.continuation_noise_seed, seed
        )
        recovery_noise = noise_base_for_seed(
            args.recovery_noise_seed, seed
        )
        recovery_anchor_noise = noise_base_for_seed(
            args.recovery_anchor_noise_seed, seed
        )
        branch_count = (
            1 + args.full_count + args.pair_count + args.remaining_count
        )
        relative = Path("groups") / (
            f"chain3-seed{seed}-srcbase{source_noise}-prop{proposal_noise}-"
            f"ar{args.recovery_anchor_blocks}"
            f"-r{args.recovery_warmup_blocks}c{args.recovery_candidates}"
            f"-rpn{recovery_noise}-"
            f"sa{args.source_attempts}-"
            f"nbase{branch_count}-f{args.full_count}p{args.pair_count}"
            f"a{args.remaining_count}-m{model_key}"
            f"-c{contract_hash[:10]}.pt"
        )
        destination = output / relative
        split = split_by_seed[seed]
        attempt_key = f"seed{seed}-c{contract_hash[:10]}"
        if destination.exists() and not args.overwrite:
            existing_group = load_branch_group(destination)
            existing_hash = existing_group["provenance"].get(
                "collection_contract_sha256"
            )
            if existing_hash != contract_hash:
                raise RuntimeError(
                    f"existing group contract {existing_hash!r} does not "
                    f"match requested {contract_hash!r}: {destination}"
                )
            print(f"[skip] seed {seed}: {destination}", flush=True)
            paths_by_split[split].append(str(relative))
            path_metadata[str(relative)] = {
                "source_trajectory_id": existing_group[
                    "source_trajectory_id"
                ],
                "snapshot_id": existing_group["snapshot_id"],
                "split": split,
                "source_seed": seed,
                "collection_contract_sha256": contract_hash,
            }
            collection_attempts[attempt_key] = {
                "status": "existing_valid_group",
                "source_seed": seed,
                "split": split,
                "path": str(relative),
                "collection_contract_sha256": contract_hash,
            }
            continue

        env = make_chain_env("chain3_lr2")
        try:
            source_search_trace = []
            context = None
            for source_attempt in range(args.source_attempts):
                attempt_noise = (
                    source_noise
                    + source_attempt * args.source_attempt_stride
                )
                context = rollout_to_stall(
                    runner,
                    env,
                    seed=seed,
                    trigger_completed=2,
                    policy_noise_seed=attempt_noise,
                    execution_horizon=args.execution_horizon,
                )
                if context is not None:
                    source_search_trace.append(
                        {
                            "attempt": source_attempt,
                            "policy_noise_seed": attempt_noise,
                            "status": "selected_2_of_3_stall",
                            "decision_index": context.decision_index,
                        }
                    )
                    break
                final_atoms = goal_atoms(env)
                final_bits = predicate_bits(env, final_atoms).tolist()
                source_search_trace.append(
                    {
                        "attempt": source_attempt,
                        "policy_noise_seed": attempt_noise,
                        "status": (
                            "full_goal_completed"
                            if all(final_bits)
                            else "stall_not_reached"
                        ),
                        "final_bits": final_bits,
                    }
                )
            if context is None:
                print(
                    f"[miss] seed {seed}: no source attempt reached the "
                    "2/3 stall",
                    flush=True,
                )
                collection_attempts[attempt_key] = {
                    "status": "source_state_unavailable",
                    "source_seed": seed,
                    "split": split,
                    "source_search_trace": source_search_trace,
                    "collection_contract_sha256": contract_hash,
                }
                continue
            context.source_trajectory_id = (
                f"chain3_lr2-full-seed{seed}-"
                f"src{context.source_policy_noise_seed}"
            )
            atoms = context.atoms
            if len(atoms) != 3:
                raise RuntimeError(f"expected three Chain-3 atoms, got {atoms}")

            object_bodies = discover_object_bodies(env)
            remaining_instruction = SUBGOAL_PHRASE[atoms[2][1]]
            recovery_diagnostics: list[dict] = []
            if args.recovery_anchor_blocks:
                context = advance_recovery_context(
                    runner,
                    env,
                    context,
                    CHAIN3_TARGET_SUPPORT_INSTRUCTION,
                    behavior_prompt_id="support_pair_recovery",
                    blocks=args.recovery_anchor_blocks,
                    behavior_noise_seed=recovery_anchor_noise,
                    full_reference_noise_seed=context.next_policy_noise_seed,
                    target_body_name=object_bodies[atoms[2][1]],
                    candidates_per_block=args.recovery_candidates,
                    stop_eef_distance=args.recovery_anchor_stop_distance,
                    execution_horizon=args.execution_horizon,
                    diagnostics=recovery_diagnostics,
                )
                if context is None:
                    reason = (
                        recovery_diagnostics[-1]["status"]
                        if recovery_diagnostics
                        else "unknown"
                    )
                    print(
                        f"[miss] seed {seed}: anchor recovery unavailable "
                        f"({reason})",
                        flush=True,
                    )
                    collection_attempts[attempt_key] = {
                        "status": "anchor_recovery_unavailable",
                        "source_seed": seed,
                        "split": split,
                        "source_search_trace": source_search_trace,
                        "recovery_diagnostics": recovery_diagnostics,
                        "collection_contract_sha256": contract_hash,
                    }
                    continue
                if context.mined_successor is None:
                    anchor_distance = float(
                        context.recovery_trace[-1]["chosen_eef_distance"]
                    )
                    if (
                        anchor_distance
                        > args.recovery_anchor_admission_distance
                    ):
                        print(
                            f"[miss] seed {seed}: anchor ended at "
                            f"{anchor_distance:.3f} m, outside "
                            f"{args.recovery_anchor_admission_distance:.3f} "
                            "m admission gate",
                            flush=True,
                        )
                        collection_attempts[attempt_key] = {
                            "status": "anchor_distance_gate_failed",
                            "source_seed": seed,
                            "split": split,
                            "source_search_trace": source_search_trace,
                            "recovery_diagnostics": recovery_diagnostics,
                            "anchor_final_distance": anchor_distance,
                            "anchor_distance_gate": (
                                args.recovery_anchor_admission_distance
                            ),
                            "collection_contract_sha256": contract_hash,
                        }
                        continue
                else:
                    print(
                        f"[mine] seed {seed}: preserving a known target-effect "
                        "anchor action as a sibling branch",
                        flush=True,
                    )
            if context.mined_successor is None:
                context = advance_recovery_context(
                    runner,
                    env,
                    context,
                    remaining_instruction,
                    behavior_prompt_id="remaining_atomic_recovery",
                    blocks=args.recovery_warmup_blocks,
                    behavior_noise_seed=recovery_noise,
                    full_reference_noise_seed=context.next_policy_noise_seed,
                    target_body_name=object_bodies[atoms[2][1]],
                    candidates_per_block=args.recovery_candidates,
                    stop_eef_distance=args.recovery_stop_distance,
                    execution_horizon=args.execution_horizon,
                    diagnostics=recovery_diagnostics,
                )
                if context is None:
                    reason = (
                        recovery_diagnostics[-1]["status"]
                        if recovery_diagnostics
                        else "unknown"
                    )
                    print(
                        f"[miss] seed {seed}: recovery warmup unavailable "
                        f"({reason})",
                        flush=True,
                    )
                    collection_attempts[attempt_key] = {
                        "status": "target_recovery_unavailable",
                        "source_seed": seed,
                        "split": split,
                        "source_search_trace": source_search_trace,
                        "recovery_diagnostics": recovery_diagnostics,
                        "collection_contract_sha256": contract_hash,
                    }
                    continue
                if (
                    context.mined_successor is None
                    and args.recovery_warmup_blocks > 0
                ):
                    target_record = context.recovery_trace[-1]
                    target_distance = float(
                        target_record["chosen_eef_distance"]
                    )
                    target_displacement = float(
                        target_record["chosen_target_displacement"]
                    )
                    if (
                        target_distance > args.recovery_stop_distance
                        and target_displacement < 0.002
                    ):
                        print(
                            f"[miss] seed {seed}: target recovery ended at "
                            f"{target_distance:.3f} m with only "
                            f"{target_displacement:.4f} m target motion",
                            flush=True,
                        )
                        collection_attempts[attempt_key] = {
                            "status": "target_effect_gate_failed",
                            "source_seed": seed,
                            "split": split,
                            "source_search_trace": source_search_trace,
                            "recovery_diagnostics": recovery_diagnostics,
                            "target_final_distance": target_distance,
                            "target_displacement": target_displacement,
                            "collection_contract_sha256": contract_hash,
                        }
                        continue
            proposals = [
                ProposalSpec(
                    proposal_prompt_id="stock_full",
                    proposal_family="stock_full",
                    goal_id="chain3_full",
                    instruction=context.full_instruction,
                    count=1,
                ),
                ProposalSpec(
                    proposal_prompt_id="full",
                    proposal_family="full",
                    goal_id="chain3_full",
                    instruction=context.full_instruction,
                    count=args.full_count,
                ),
                ProposalSpec(
                    proposal_prompt_id="exact_pair",
                    proposal_family="exact_pair",
                    goal_id="chain3_exact_pair",
                    instruction=CHAIN3_PAIR_INSTRUCTION,
                    count=args.pair_count,
                ),
                ProposalSpec(
                    proposal_prompt_id="remaining_atomic",
                    proposal_family="remaining_atomic",
                    goal_id="chain3_remaining_atomic",
                    instruction=remaining_instruction,
                    count=args.remaining_count,
                ),
            ]
            group = collect_snapshot_group(
                runner,
                env,
                context,
                proposals,
                list(object_bodies.values()),
                atoms[2],
                remaining_instruction,
                goal_object_body_name=object_bodies[atoms[2][1]],
                goal_receptacle_body_name=object_bodies["basket_1"],
                execution_horizon=args.execution_horizon,
                continuation_horizon=args.continuation_horizon,
                continuation_count=args.continuation_count,
                proposal_noise_seed=proposal_noise,
                continuation_noise_seed=continuation_noise,
            )
            group["goal_specs"]["chain3_full"]["atom_indices"] = [0, 1, 2]
            group["goal_specs"]["chain3_exact_pair"]["atoms"] = atoms[:2]
            group["goal_specs"]["chain3_exact_pair"]["atom_indices"] = [0, 1]
            group["goal_specs"]["chain3_remaining_atomic"][
                "atom_indices"
            ] = [2]
            group["object_names"] = list(object_bodies)
            group["body_names"] = list(object_bodies.values())
            group["source_seed"] = seed
            group["provenance"]["collection_contract"] = seed_contract
            group["provenance"][
                "collection_contract_sha256"
            ] = contract_hash
            group["provenance"][
                "source_search_trace"
            ] = source_search_trace
            group["provenance"][
                "recovery_diagnostics"
            ] = recovery_diagnostics
            coverage = branch_coverage_summary(group)
            if not bool(group["restore_qc"]["valid"]):
                rejected = output / "rejected" / destination.name
                save_branch_group(group, rejected)
                print(
                    f"[reject] seed {seed}: restore QC "
                    f"q={group['restore_qc']['q_linf']:.3e}, "
                    f"obj={group['restore_qc']['object_linf']:.3e} "
                    f"-> {rejected}",
                    flush=True,
                )
                collection_attempts[attempt_key] = {
                    "status": "restore_qc_rejected",
                    "source_seed": seed,
                    "split": split,
                    "source_search_trace": source_search_trace,
                    "recovery_diagnostics": recovery_diagnostics,
                    "rejected_path": str(rejected.relative_to(output)),
                    "coverage": coverage,
                    "restore_qc_summary": {
                        key: value
                        for key, value in group["restore_qc"].items()
                        if isinstance(
                            value, (str, int, float, bool, type(None))
                        )
                    },
                    "collection_contract_sha256": contract_hash,
                }
                continue
            save_branch_group(group, destination)
            paths_by_split[split].append(str(relative))
            path_metadata[str(relative)] = {
                "source_trajectory_id": group["source_trajectory_id"],
                "snapshot_id": group["snapshot_id"],
                "split": split,
                "source_seed": seed,
                "collection_contract_sha256": contract_hash,
            }
            collection_attempts[attempt_key] = {
                "status": "saved",
                "source_seed": seed,
                "split": split,
                "source_search_trace": source_search_trace,
                "recovery_diagnostics": recovery_diagnostics,
                "path": str(relative),
                "snapshot_id": group["snapshot_id"],
                "coverage": coverage,
                "collection_contract_sha256": contract_hash,
            }
            print(
                f"[saved] seed {seed}: {group['actions_norm'].shape[0]} branches "
                f"effectful={coverage['object_effectful_branches']} "
                f"eligible={coverage['positive_flow_targets']} "
                f"-> {destination}",
                flush=True,
            )
        finally:
            env.close()

    write_manifest(
        output,
        contract,
        paths_by_split,
        path_metadata,
        collection_attempts,
    )
    print(f"manifest -> {output / 'branch_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
