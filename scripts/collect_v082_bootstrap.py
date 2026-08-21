#!/usr/bin/env python3
"""Collect the fresh, source-disjoint sibling tensor bank used to train M0.

This is the mainline handoff from Action 2P's evidence to world-model
training.  It does not reuse Action-2P rows and it does not run another
promotion diagnostic.  The only admission condition is that a source yields
the already established chain1b@250 / tau=160 exact-mask anchor needed to
construct a causally valid group.

Usage::

    conda run -n vf0s python scripts/collect_v082_bootstrap.py --plan
    conda run -n vf0s python scripts/collect_v082_bootstrap.py --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm import v082_bootstrap as B  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.snapshot import restore  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS  # noqa: E402
from lcwm.v080r_panel import make_env_at  # noqa: E402
from lcwm.v081p_exec import (  # noqa: E402
    SegmentLedger,
    body_signature,
    run_source,
    select_candidates,
    verify_restore,
)


TASK = "chain1b_lr2"
DEADLINE = 250
TAU = 160
C_PREFIX = 10
H = 80
N_ALT_DRAWS = 32
N_DIVERSITY = 3
N_RANDOM = 3
CANDIDATES = 1 + N_DIVERSITY + N_RANDOM

QUEUES = {
    "train": tuple(range(3500, 3540)),
    "val": tuple(range(3540, 3554)),
    "test": tuple(range(3554, 3576)),
}
QUOTAS = {"train": 16, "val": 4, "test": 8}
REPEATS = dict(B.REPEATS)

SOURCE_CAP = sum(len(q) for q in QUEUES.values()) * DEADLINE       # 19,000
N_BRANCHES = sum(QUOTAS[s] * CANDIDATES * REPEATS[s] for s in QUOTAS)
BRANCH_CAP = N_BRANCHES * (C_PREFIX + H)                           # 27,720
CAPS = {"source": SOURCE_CAP, "branch": BRANCH_CAP}
TOTAL_CAP = SOURCE_CAP + BRANCH_CAP                                # 46,720
assert N_BRANCHES == 308 and TOTAL_CAP == 46_720

OUT_ROOT = REPO / "results" / "v082_bootstrap"
OUTCOME_KEYS = ("dmg", "succ", "dp", "ttm", "G")


def _seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _key(root: str, *parts: object) -> int:
    digest = hashlib.sha256("|".join([root, *map(str, parts)]).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _git_state() -> tuple[str, bool]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip())
    return head, dirty


def _raw_obs(env):
    return env._format_raw_obs(env._env.env._get_observations())


def build_training_pool(runner, env, anchor, root: str) -> dict:
    """Freeze the PI0 prefix tap and select 1+3+3 first-10 action siblings."""

    restore(env, anchor.snapshot)
    runner.reset()
    obs = _raw_obs(env)
    policy_obs = runner._obs_to_policy_batch(obs, env.task_description)
    prefix = prefix_forward(runner.policy, policy_obs)
    hidden = prefix.hidden[0].detach().to(device="cpu", dtype=torch.float16)
    hidden_mask = prefix.pad_masks[0].detach().to(device="cpu", dtype=torch.bool)

    ref_norm = sample_chunks(
        runner.policy,
        policy_obs,
        1,
        seed=_key(root, anchor.anchor_id, "reference"),
        prefix=prefix,
    )[0, :C_PREFIX].detach().float().cpu()
    alt_norm = sample_chunks(
        runner.policy,
        policy_obs,
        N_ALT_DRAWS,
        seed=_key(root, anchor.anchor_id, "alternatives"),
        prefix=prefix,
    )[:, :C_PREFIX].detach().float().cpu()
    del prefix

    ref_env = runner.chunk_to_env(ref_norm)
    alt_env = [runner.chunk_to_env(chunk) for chunk in alt_norm]
    seen: dict[bytes, int] = {}
    unique_idx: list[int] = []
    for i, chunk in enumerate(alt_env):
        key = np.round(chunk, 4).tobytes()
        if key not in seen:
            seen[key] = i
            unique_idx.append(i)
    if len(unique_idx) < N_DIVERSITY + N_RANDOM:
        raise RuntimeError(
            f"{anchor.anchor_id}: only {len(unique_idx)} unique alternatives"
        )
    selection = select_candidates(
        {"alternatives": alt_env, "unique_idx": unique_idx},
        root,
        anchor.anchor_id,
    )
    indices = selection["diversity"] + selection["random"]
    ids = ["reference", "div0", "div1", "div2", "rand0", "rand1", "rand2"]
    actions_norm = torch.stack([ref_norm, *[alt_norm[i] for i in indices]])
    actions_env = torch.from_numpy(np.stack(
        [ref_env, *[alt_env[i] for i in indices]]
    )).float()
    restore(env, anchor.snapshot)
    anchor_sig = torch.from_numpy(body_signature(env)).float()
    if anchor_sig.shape != (24,):
        raise RuntimeError(
            f"{anchor.anchor_id}: expected 24-D body signature, got {anchor_sig.shape}"
        )
    return {
        "prefix_hidden": hidden,
        "prefix_mask": hidden_mask,
        "anchor_signature": anchor_sig,
        "actions_norm": actions_norm,
        "actions_env": actions_env,
        "candidate_ids": ids,
        "is_reference": torch.tensor(
            [True, False, False, False, False, False, False], dtype=torch.bool
        ),
        "n_unique": len(unique_idx),
        "selection": selection,
    }


def execute_branch(
    runner,
    env,
    anchor,
    actions_env: np.ndarray,
    candidate_id: str,
    repeat: int,
    continuation_seed: int,
    ledger: SegmentLedger,
) -> dict:
    """Execute one first-10 sibling and label the following 80-step future."""

    segment_id = f"{anchor.anchor_id}-{candidate_id}-r{repeat}"
    ledger.open_segment(
        segment_id,
        "branch",
        C_PREFIX + H,
        anchor_id=anchor.anchor_id,
        candidate_id=candidate_id,
        repeat=repeat,
    )
    restore(env, anchor.snapshot)
    runner.reset()
    restore_ok, restored_hash = verify_restore(env, anchor)
    if not restore_ok:
        raise RuntimeError(
            f"{anchor.anchor_id}: restore hash {restored_hash} != {anchor.content_hash}"
        )

    subgoals = V080_TASKS[TASK]["ordered_subgoals"]
    automaton = (anchor.source_automaton.fork()
                 if anchor.source_automaton is not None
                 else GoalAutomaton(subgoals))
    if anchor.source_automaton is None:
        automaton.start(env)
    atoms = goal_atoms(env)
    automaton.evaluate(env, TAU)
    anchor_sig = body_signature(env)
    obs = _raw_obs(env)
    instruction = env.task_description
    t, done, success_step = TAU, False, None

    for action in actions_env:
        obs, _reward, terminated, truncated, info = env.step(action)
        t += 1
        done = bool(terminated or truncated)
        if success_step is None and bool(info.get("is_success", False)):
            success_step = t
        if done:
            break
    c_eff = t - TAU
    if c_eff != C_PREFIX:
        raise RuntimeError(
            f"{anchor.anchor_id}/{candidate_id}/r{repeat}: only {c_eff}/10 actions executed"
        )
    automaton.evaluate(env, t)
    if success_step is None and predicate_bits(env, atoms).all():
        success_step = t
    post_delta = body_signature(env) - anchor_sig
    immediate_events = {
        int(i): int(step) for i, step in automaton.events_achieved.items()
        if TAU < int(step) <= t
    }
    immediate_bits = np.asarray(
        [bool(immediate_events), automaton.damage_unrecovered() > 0],
        dtype=np.bool_,
    )

    runner.reset()
    _seed_all(continuation_seed)
    continuation_steps = 0
    while not done and continuation_steps < H and t < DEADLINE:
        obs, _reward, terminated, truncated, info = env.step(
            runner.select_action(obs, instruction)
        )
        t += 1
        continuation_steps += 1
        done = bool(terminated or truncated)
        if success_step is None and bool(info.get("is_success", False)):
            success_step = t
        if t % 10 == 0 or done:
            automaton.evaluate(env, t)
            if success_step is None and predicate_bits(env, atoms).all():
                success_step = t
    if t > DEADLINE:
        raise RuntimeError(f"post-deadline step: {t}>{DEADLINE}")
    outcome = B.continuation_outcome(
        {int(i): int(step) for i, step in automaton.events_achieved.items()},
        success_step,
        automaton.damage_unrecovered(),
        TAU,
        C_PREFIX,
        H,
    )
    actual_steps = C_PREFIX + continuation_steps
    ledger.close_segment(
        segment_id,
        "branch",
        C_PREFIX + H,
        actual_steps,
        anchor_id=anchor.anchor_id,
        candidate_id=candidate_id,
        repeat=repeat,
        continuation_seed=continuation_seed,
        termination="success_or_done" if done else "horizon",
    )
    return {
        "post_prefix_delta": torch.from_numpy(post_delta).float(),
        "immediate_bits": torch.from_numpy(immediate_bits),
        "outcome_h80": torch.tensor(
            [outcome[key] for key in OUTCOME_KEYS], dtype=torch.float32
        ),
        "steps": actual_steps,
        "success_step": success_step,
    }


def make_group(runner, env, anchor, split: str, root: str,
               ledger: SegmentLedger) -> tuple[dict, dict]:
    pool = build_training_pool(runner, env, anchor, root)
    repeats = REPEATS[split]
    delta = torch.zeros(CANDIDATES, repeats, 24)
    bits = torch.zeros(CANDIDATES, repeats, 2, dtype=torch.bool)
    outcomes = torch.zeros(CANDIDATES, repeats, 5)
    branch_rows = []
    continuation_keys = [
        _key(root, anchor.anchor_id, "continuation", r) for r in range(repeats)
    ]
    for candidate_index, candidate_id in enumerate(pool["candidate_ids"]):
        action = pool["actions_env"][candidate_index].numpy()
        for repeat, continuation_seed in enumerate(continuation_keys):
            row = execute_branch(
                runner,
                env,
                anchor,
                action,
                candidate_id,
                repeat,
                continuation_seed,
                ledger,
            )
            delta[candidate_index, repeat] = row["post_prefix_delta"]
            bits[candidate_index, repeat] = row["immediate_bits"]
            outcomes[candidate_index, repeat] = row["outcome_h80"]
            branch_rows.append({
                "candidate_id": candidate_id,
                "repeat": repeat,
                "continuation_seed": continuation_seed,
                "steps": row["steps"],
                "success_step": row["success_step"],
            })
    group = {
        "schema": B.SCHEMA,
        "anchor_id": anchor.anchor_id,
        "source_id": f"{TASK}:{anchor.seed}",
        "split": split,
        "role": B.SPLIT_ROLES[split],
        "prefix_hidden": pool["prefix_hidden"],
        "prefix_mask": pool["prefix_mask"],
        "anchor_signature": pool["anchor_signature"],
        "actions_norm": pool["actions_norm"],
        "actions_env": pool["actions_env"],
        "candidate_ids": pool["candidate_ids"],
        "is_reference": pool["is_reference"],
        "post_prefix_delta": delta,
        "immediate_bits": bits,
        "outcome_h80": outcomes,
        "repeat_mask": torch.ones(CANDIDATES, repeats, dtype=torch.bool),
        "provenance": {
            "task": TASK,
            "source_seed": anchor.seed,
            "tau": TAU,
            "c_eff": C_PREFIX,
            "continuation_horizon": H,
            "anchor_content_hash": anchor.content_hash,
            "pool_unique": pool["n_unique"],
            "pool_selection": pool["selection"],
            "continuation_keys": continuation_keys,
        },
    }
    B.validate_bootstrap_group(group)
    return group, {"anchor_id": anchor.anchor_id, "split": split,
                   "branches": branch_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.plan == args.run:
        parser.error("pass exactly one of --plan or --run")

    head, dirty = _git_state()
    root = hashlib.sha256(
        f"v082-bootstrap-v1|{head}|{QUEUES}|{QUOTAS}|{REPEATS}".encode()
    ).hexdigest()
    plan = {
        "schema": "v082_bootstrap_manifest_v1",
        "task": TASK,
        "deadline": DEADLINE,
        "tau": TAU,
        "c_prefix": C_PREFIX,
        "continuation_horizon": H,
        "queues": {key: list(value) for key, value in QUEUES.items()},
        "quotas": QUOTAS,
        "repeats": REPEATS,
        "candidates_per_anchor": CANDIDATES,
        "source_cap": SOURCE_CAP,
        "branch_cap": BRANCH_CAP,
        "total_cap": TOTAL_CAP,
        "git_head": head,
        "git_dirty": dirty,
        "rng_root": root,
        "old_v081p_rows_used": False,
    }
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    if dirty:
        raise SystemExit("commit the collector/trainer before the environment run")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps({**plan, "status": "RUNNING"}, indent=2))
    ledger = SegmentLedger(out / "segment_ledger.jsonl", CAPS, action_root=out)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner

    runner = Pi05Runner(
        model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C_PREFIX
    )
    env = make_env_at(TASK, DEADLINE)
    source_rows: list[dict] = []
    anchors: list[tuple[str, object]] = []

    for split in ("train", "val", "test"):
        kept = 0
        for seed in QUEUES[split]:
            source = run_source(runner, env, seed, ledger)
            anchor = source["anchor"]
            eligible = bool(
                source["failure_at_L"] and anchor is not None and anchor.mask_ok
            )
            source_rows.append({
                "split": split,
                "role": B.SPLIT_ROLES[split],
                "seed": seed,
                "steps": source["steps"],
                "success_step": source["success_step"],
                "failure_at_L": source["failure_at_L"],
                "eligible": eligible,
                "anchor_id": anchor.anchor_id if anchor else None,
                "mask_why": anchor.mask_why if anchor else ["no tau snapshot"],
            })
            print(
                f"source split={split} seed={seed} "
                f"failure={source['failure_at_L']} eligible={eligible} "
                f"quota={kept}/{QUOTAS[split]}",
                flush=True,
            )
            if eligible:
                anchors.append((split, anchor))
                kept += 1
            if kept == QUOTAS[split]:
                break
        if kept != QUOTAS[split]:
            raise SystemExit(
                f"insufficient {split} anchors: {kept}/{QUOTAS[split]} "
                f"within fixed source queue"
            )
    (out / "sources.json").write_text(json.dumps(source_rows, indent=2))

    groups: list[dict] = []
    branch_index: list[dict] = []
    for index, (split, anchor) in enumerate(anchors, start=1):
        print(
            f"group {index}/{sum(QUOTAS.values())}: "
            f"{split} {anchor.anchor_id}",
            flush=True,
        )
        group, branches = make_group(runner, env, anchor, split, root, ledger)
        B.save_bootstrap_group(group, out / "groups" / f"{split}_{anchor.seed}.pt")
        groups.append(group)
        branch_index.append(branches)

    B.validate_groups(groups)
    if {split: sum(g["split"] == split for g in groups) for split in QUOTAS} != QUOTAS:
        raise RuntimeError("final split quotas do not match the fixed 16/4/8 contract")
    torch.save(
        {
            "schema": "v082_bootstrap_bank_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "groups": groups,
            "manifest": plan,
        },
        out / "groups.pt",
    )
    (out / "branch_index.json").write_text(json.dumps(branch_index, indent=2))
    spend = {
        "by_cap_line": ledger.spent,
        "actual_total": ledger.total,
        "hard_cap": TOTAL_CAP,
        "source_attempts": len(source_rows),
        "groups": len(groups),
        "branches": N_BRANCHES,
    }
    (out / "SPEND.json").write_text(json.dumps(spend, indent=2))
    manifest_path.write_text(json.dumps(
        {**plan, "status": "COMPLETE", "output": str(out), "spend": spend},
        indent=2,
    ))
    print(f"COMPLETE: {out / 'groups.pt'}", flush=True)
    print(json.dumps(spend, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
