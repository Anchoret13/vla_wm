#!/usr/bin/env python
"""Collect contact-proximal on-policy branch groups (chain3_contact_v1).

Pre-registered 2026-07-24 (plan_and_progress/2026-07-24.md, execution
section). Motivation: the stall+recovery collector produced no target-effect
groups on fresh seeds (16 attempts -> 3 source episodes), while full-prompt
rollouts reliably manipulate the first two chain3 goal objects (15/15 each in
chain_exam_v2). This collector snapshots the SAME on-policy full-prompt
rollout at the first decision boundary where the EEF is within
--contact-distance of a goal object whose atom is still false, then collects
the standard 13-branch sibling group there with `collect_snapshot_group`
unchanged. No recovery bridge. The dense-progress readout targets the
contacted object; raw decision-boundary observations are stored so crossed
languages can later be recomputed over the complete history (audit
correction 5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import collect_chain3_pilot as pilot  # noqa: E402  (shared contract helpers)

# v1_1 (AMENDMENT 1, 2026-07-24.md): contact-contract admission is
# structural_valid AND object_resolved. The q channel's resolved flag is
# preserved per group and gates q-effect metrics downstream; both noise-ratio
# thresholds are unchanged. Rationale: contact states replay with q noise
# ~1e-3 (vs 1e-5 at stall states) while the object channel replays exactly,
# and the group-level AND-gate was discarding perfectly object-resolved
# groups — the data §15.10 explicitly demands.
COLLECTION_CONTRACT_VERSION = "chain3_contact_v1_1_20260724"
ADMISSION_RULE = "structural_valid_and_object_resolved"
COLLECTOR_SOURCE_FILES = pilot.COLLECTOR_SOURCE_FILES + (
    "scripts/collect_chain3_contact.py",
)
DEFAULT_OUTPUT = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/chain_branches_v1_1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--full-count", type=int, default=4)
    parser.add_argument("--pair-count", type=int, default=4)
    parser.add_argument("--remaining-count", type=int, default=4)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--contact-distance", type=float, default=0.05)
    parser.add_argument("--contact-skip", type=int, default=0)
    parser.add_argument("--source-attempts", type=int, default=2)
    parser.add_argument("--source-attempt-stride", type=int, default=100)
    parser.add_argument("--continuation-horizon", type=int, default=160)
    parser.add_argument("--continuation-count", type=int, default=4)
    parser.add_argument("--source-noise-seed", type=int, default=20_000)
    parser.add_argument("--proposal-noise-seed", type=int, default=30_000)
    parser.add_argument("--continuation-noise-seed", type=int, default=40_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collector_source_sha256() -> str:
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
    return {
        "version": COLLECTION_CONTRACT_VERSION,
        "collector_source_sha256": collector_source_sha256(),
        "task": "chain3_lr2",
        "source_seed": seed,
        "split": pilot.assign_splits([seed])[seed],
        "base_model_id": base_model_id,
        "num_inference_steps": num_inference_steps,
        "snapshot_mode": "on_policy_contact",
        "admission_rule": ADMISSION_RULE,
        "contact": {
            "distance": args.contact_distance,
            "skip_matches": args.contact_skip,
            "trigger": (
                "EEF within distance of the root body of the nearest goal "
                "object with a false atom, evaluated at decision boundaries"
            ),
        },
        "proposal_counts": {
            "stock_full": 1,
            "full": args.full_count,
            "exact_pair": args.pair_count,
            "remaining_atomic": args.remaining_count,
        },
        "proposal_instructions": {
            "exact_pair": pilot.CHAIN3_PAIR_INSTRUCTION,
            "remaining_atomic": "SUBGOAL_PHRASE[contact object]",
        },
        "execution_horizon": args.execution_horizon,
        "source_search": {
            "attempts": args.source_attempts,
            "attempt_stride": args.source_attempt_stride,
        },
        "continuation": {
            "horizon": args.continuation_horizon,
            "count": args.continuation_count,
        },
        "stores_raw_history_observations": True,
        "noise_seeds": {
            "source": pilot.noise_base_for_seed(args.source_noise_seed, seed),
            "proposal": pilot.noise_base_for_seed(
                args.proposal_noise_seed, seed
            ),
            "continuation": pilot.noise_base_for_seed(
                args.continuation_noise_seed, seed
            ),
        },
    }


def resolved_contract(args: argparse.Namespace) -> dict:
    if min(args.full_count, args.pair_count, args.remaining_count) < 0:
        raise ValueError("proposal counts must be nonnegative")
    if args.execution_horizon != 10:
        raise ValueError("LC-Flow v0 uses an exact 10-action commitment")
    if args.contact_distance <= 0:
        raise ValueError("--contact-distance must be positive")
    if args.contact_skip < 0:
        raise ValueError("--contact-skip must be nonnegative")
    if not 1 <= args.source_attempts <= 10:
        raise ValueError("--source-attempts must be in [1,10]")
    if args.source_attempt_stride < 100:
        raise ValueError("--source-attempt-stride must be at least 100")
    if (args.source_attempts - 1) * args.source_attempt_stride >= 1_000:
        raise ValueError(
            "source attempts would cross the next seed's noise block"
        )
    if args.continuation_horizon < 1:
        raise ValueError("--continuation-horizon must be positive")
    branch_count = (
        1 + args.full_count + args.pair_count + args.remaining_count
    )
    if not 0 <= args.continuation_count <= branch_count:
        raise ValueError("--continuation-count must be within the branch pool")
    split_by_seed = pilot.assign_splits(args.seeds)
    return {
        "contract_version": COLLECTION_CONTRACT_VERSION,
        "collector_source_sha256": collector_source_sha256(),
        "task": "chain3_lr2",
        "output": str(args.output.resolve()),
        "source_seeds": list(dict.fromkeys(args.seeds)),
        "split_by_seed": {
            str(seed): split_by_seed[seed] for seed in split_by_seed
        },
        "snapshot_mode": "on_policy_contact",
        "admission_rule": ADMISSION_RULE,
        "contact_distance": args.contact_distance,
        "contact_skip": args.contact_skip,
        "base_branches_per_group": branch_count,
        "proposal_counts": {
            "stock_full": 1,
            "full": args.full_count,
            "exact_pair": args.pair_count,
            "remaining_atomic": args.remaining_count,
        },
        "execution_horizon": args.execution_horizon,
        "source_attempts": args.source_attempts,
        "source_attempt_stride": args.source_attempt_stride,
        "continuation_horizon": args.continuation_horizon,
        "continuation_count": args.continuation_count,
        "stores_raw_history_observations": True,
        "split_rule": "seed_mod_10: 0..7=train, 8=dev, 9=test",
        "noise_rule": "base + (source_seed - 1000) * 1000",
        "noise_seed_bases": {
            "source": args.source_noise_seed,
            "proposal": args.proposal_noise_seed,
            "continuation": args.continuation_noise_seed,
        },
    }


def main() -> None:
    args = parse_args()
    contract = resolved_contract(args)
    print(json.dumps(contract, indent=2), flush=True)
    if args.dry_run:
        return

    from lcwm.branch_collect import (
        ProposalSpec,
        collect_snapshot_group,
        rollout_to_contact,
    )
    from lcwm.branch_data import load_branch_group, save_branch_group
    from lcwm.chassis import Pi05Runner
    from lcwm.loho import SUBGOAL_PHRASE, make_chain_env
    from lcwm.probe_data import discover_object_bodies

    output = args.output.resolve()
    groups_dir = output / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    split_by_seed = pilot.assign_splits(args.seeds)
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
        contract_hash = pilot.contract_sha256(seed_contract)
        source_noise = pilot.noise_base_for_seed(args.source_noise_seed, seed)
        proposal_noise = pilot.noise_base_for_seed(
            args.proposal_noise_seed, seed
        )
        continuation_noise = pilot.noise_base_for_seed(
            args.continuation_noise_seed, seed
        )
        branch_count = (
            1 + args.full_count + args.pair_count + args.remaining_count
        )
        relative = Path("groups") / (
            f"chain3ct-seed{seed}-skip{args.contact_skip}-"
            f"cd{round(args.contact_distance * 1000)}mm-"
            f"src{source_noise}-prop{proposal_noise}-"
            f"n{branch_count}-f{args.full_count}p{args.pair_count}"
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
                "source_trajectory_id": existing_group["source_trajectory_id"],
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
            contact_info = None
            contact_diagnostics: list[dict] = []
            for source_attempt in range(args.source_attempts):
                attempt_noise = (
                    source_noise
                    + source_attempt * args.source_attempt_stride
                )
                attempt_diagnostics: list[dict] = []
                result = rollout_to_contact(
                    runner,
                    env,
                    seed=seed,
                    contact_distance=args.contact_distance,
                    skip_matches=args.contact_skip,
                    policy_noise_seed=attempt_noise,
                    execution_horizon=args.execution_horizon,
                    diagnostics=attempt_diagnostics,
                )
                if result is not None:
                    context, contact_info = result
                    contact_diagnostics = attempt_diagnostics
                    source_search_trace.append(
                        {
                            "attempt": source_attempt,
                            "policy_noise_seed": attempt_noise,
                            "status": "selected_contact_state",
                            "decision_index": context.decision_index,
                            "contact_object": contact_info["contact_object"],
                            "contact_distance": (
                                contact_info["contact_distance"]
                            ),
                        }
                    )
                    break
                minimum = min(
                    (
                        record["nearest_distance"]
                        for record in attempt_diagnostics
                        if record["nearest_incomplete_object"] is not None
                    ),
                    default=float("inf"),
                )
                source_search_trace.append(
                    {
                        "attempt": source_attempt,
                        "policy_noise_seed": attempt_noise,
                        "status": "no_contact_state",
                        "min_nearest_distance": minimum,
                        "decisions": len(attempt_diagnostics),
                        "final_bits": (
                            attempt_diagnostics[-1]["bits"]
                            if attempt_diagnostics
                            else None
                        ),
                    }
                )
            if context is None:
                print(
                    f"[miss] seed {seed}: no contact state within "
                    f"{args.contact_distance:.3f} m",
                    flush=True,
                )
                collection_attempts[attempt_key] = {
                    "status": "contact_state_unavailable",
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
                raise RuntimeError(
                    f"expected three Chain-3 atoms, got {atoms}"
                )
            object_bodies = discover_object_bodies(env)
            contact_object = contact_info["contact_object"]
            contact_index = next(
                index
                for index, atom in enumerate(atoms)
                if atom[1] == contact_object
            )
            remaining_instruction = SUBGOAL_PHRASE[contact_object]

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
                    instruction=pilot.CHAIN3_PAIR_INSTRUCTION,
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
                atoms[contact_index],
                remaining_instruction,
                goal_object_body_name=object_bodies[contact_object],
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
            ] = [contact_index]
            group["object_names"] = list(object_bodies)
            group["body_names"] = list(object_bodies.values())
            group["source_seed"] = seed
            group["history_observations"] = context.history_observations
            group["provenance"]["collection_contract"] = seed_contract
            group["provenance"]["collection_contract_sha256"] = contract_hash
            group["provenance"]["source_search_trace"] = source_search_trace
            group["provenance"]["contact_info"] = contact_info
            group["provenance"]["contact_diagnostics"] = contact_diagnostics
            coverage = pilot.branch_coverage_summary(group)
            admitted = bool(
                group["restore_qc"]["structural_valid"]
                and group["restore_qc"]["object_resolved"]
            )
            if not admitted:
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
                "path": str(relative),
                "snapshot_id": group["snapshot_id"],
                "contact_info": contact_info,
                "coverage": coverage,
                "collection_contract_sha256": contract_hash,
            }
            print(
                f"[saved] seed {seed}: "
                f"{group['actions_norm'].shape[0]} branches "
                f"contact={contact_object}@"
                f"{contact_info['contact_distance']:.3f}m "
                f"decision={contact_info['decision_index']} "
                f"effectful={coverage['object_effectful_branches']} "
                f"eligible={coverage['positive_flow_targets']} "
                f"q_resolved={group['restore_qc']['q_resolved']} "
                f"obj_ratio={group['restore_qc']['object_effect_noise_ratio']:.1f} "
                f"-> {destination}",
                flush=True,
            )
        finally:
            env.close()

    pilot.write_manifest(
        output,
        contract,
        paths_by_split,
        path_metadata,
        collection_attempts,
    )
    print(f"manifest -> {output / 'branch_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
