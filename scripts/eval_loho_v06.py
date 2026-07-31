#!/usr/bin/env python
"""V6.6 — public LIBERO-LoHo N=1 development matrix for the v0.6 arms.

Arms (registered):
  stock                zero bias (same seeded LC code path — parity)
  v06_gt_current       v06_gt W_z, reset/current z each decision
  v06_gt_recurrent     SAME v06_gt W_z, carried z
  v06_recurrent_random v06_random W_z, carried z
  v06_recurrent_wm     v06_wm W_z, carried z

Contract: dev seeds {1600,1610,1620,1630,1640} (sealed 1700s untouched);
noise 20e6 + public_task_index*2e6 + env_seed*1e3 + decision, shared
bit-exactly across arms; full composite prompts, N=1, chunk 50 / execute
10; per-run immutable manifest; hash-verified resume; hash-permutation
interleave; automaton-valid metrics (valid Q, ordered prefix,
invalidation) + terminal-conjunction SR primary.
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
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
ROOT = REPO_ROOT / "results" / "libero_loho_public_v1" / "v06_eval"
WM_CKPT = (REPO_ROOT / "results" / "libero_loho_public_v1" / "v06_wm"
           / "checkpoint_final.pt")
POLICY_DIR = REPO_ROOT / "results" / "libero_loho_public_v1" / "v06_policy"
ARM_SPEC = {
    "stock": None,
    "v06_gt_current": ("v06_gt", "reset"),
    "v06_gt_recurrent": ("v06_gt", "recurrent"),
    "v06_recurrent_random": ("v06_random", "recurrent"),
    "v06_recurrent_wm": ("v06_wm", "recurrent"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_seed(task: str, env_seed: int, decision: int) -> int:
    return (NOISE_BASE + TASK_ORDER.index(task) * 2_000_000
            + env_seed * 1_000 + decision)


def arm_order(task: str, env_seed: int, arms: list[str]) -> list[str]:
    return sorted(arms, key=lambda arm: hashlib.sha256(
        f"{task}|{env_seed}|{arm}".encode()).hexdigest())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arms", nargs="+", required=True,
                        choices=sorted(ARM_SPEC))
    parser.add_argument("--tasks", nargs="+", default=TASK_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[1600, 1610, 1620, 1630, 1640])
    parser.add_argument("--panel", default="development")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import sample_chunks_lc
    from lcwm.loho_public import load_public_tasks, make_public_env
    from lcwm.sampler import prefix_forward
    from lcwm.task_automaton import GoalAutomaton, terminal_success
    from lcwm.v06_model import V06State

    panel_dir = ROOT / args.panel
    panel_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "traces").mkdir(exist_ok=True)
    records_path = panel_dir / "records.jsonl"
    tasks_spec = load_public_tasks()

    wz_files = {arm: POLICY_DIR / ARM_SPEC[arm][0] / "wz_selected.pt"
                for arm in args.arms if ARM_SPEC[arm]}
    manifest = {
        "schema": "v06_eval_v1", "run_id": args.run_id,
        "noise_contract": ("20e6 + public_task_index*2e6 + env_seed*1e3 "
                           "+ decision; shared across arms; all arms via "
                           "sample_chunks_lc (stock bias=0)"),
        "arms": sorted(args.arms),
        "wz_hashes": {a: sha256_file(p) for a, p in wz_files.items()},
        "wm_checkpoint_sha256": sha256_file(WM_CKPT),
        "tasks": args.tasks, "seeds": args.seeds,
        "code_files": {f: sha256_file(REPO_ROOT / f) for f in (
            "scripts/eval_loho_v06.py", "lcwm/v06_model.py",
            "lcwm/task_automaton.py", "lcwm/loho_public.py")},
    }
    payload = json.dumps(manifest, sort_keys=True)
    manifest_hash = hashlib.sha256(payload.encode()).hexdigest()
    manifest["manifest_sha256"] = manifest_hash
    mf = ROOT / f"run_manifest_{args.run_id}.json"
    if mf.exists():
        assert json.loads(mf.read_text())["manifest_sha256"] == \
            manifest_hash, "manifest hash mismatch on resume"
    else:
        mf.write_text(json.dumps(manifest, indent=2))
    done_keys = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_keys.add((r["run_id"], r["task"], r["seed"],
                               r["arm"]))
        if not args.resume:
            assert not any(k[0] == args.run_id for k in done_keys), \
                "records exist for this run_id; use --resume"

    runner = Pi05Runner(suite_name="libero_10")
    v06 = V06State().to(device)
    v06.load_state_dict(torch.load(WM_CKPT,
                                   weights_only=False)["model"])
    v06.eval()
    zero_bias = torch.zeros(1, 1024, device=device)

    def load_arm(arm):
        if ARM_SPEC[arm] is None:
            return None, None
        wz = torch.load(wz_files[arm], weights_only=False)
        assert wz["wm_checkpoint_sha256"] == \
            manifest["wm_checkpoint_sha256"]
        v06.w_z.load_state_dict(wz["w_z"])
        return ARM_SPEC[arm][1], wz["step"]

    for task in args.tasks:
        subgoals = tasks_spec[task]["ordered_subgoals"]
        for env_seed in args.seeds:
            for arm in arm_order(task, env_seed, list(args.arms)):
                key = (args.run_id, task, env_seed, arm)
                if key in done_keys:
                    continue
                started = walltime.time()
                state_mode, _step = load_arm(arm)
                env = make_public_env(task, EPISODE_LENGTH[task])
                try:
                    runner.reset()
                    obs, _ = env.reset(seed=env_seed)
                    automaton = GoalAutomaton(list(subgoals))
                    automaton.start(env)
                    automaton.evaluate(env, 0)
                    instruction = env.task_description
                    z = None
                    actions_env, used_seeds = [], []
                    t, decision = 0, 0
                    term = trunc = False
                    while t < env.episode_length:
                        batch = runner._obs_to_policy_batch(
                            obs, instruction)
                        prefix = prefix_forward(runner.policy, batch)
                        if ARM_SPEC[arm] is None:
                            bias = zero_bias
                        else:
                            h = prefix.hidden.float()
                            m = prefix.pad_masks.bool()
                            if state_mode == "reset" or z is None:
                                z_now = v06.initial_state(h, m)
                            else:
                                z_now = v06.step(
                                    z, prev_a, h, m,
                                    action_mask=prev_m)
                            if state_mode == "recurrent":
                                z = z_now
                            bias = v06.policy_bias(z_now)
                        ns = noise_seed(task, env_seed, decision)
                        chunk = sample_chunks_lc(
                            runner.policy, batch, bias, n=1, seed=ns,
                            prefix=prefix)
                        used_seeds.append(ns)
                        executed = 0
                        for a in runner.chunk_to_env(chunk[:, :10]):
                            obs, _r, term, trunc, _i = env.step(a)
                            actions_env.append(np.asarray(a))
                            t += 1
                            executed += 1
                            if term or trunc:
                                break
                        automaton.evaluate(env, t)
                        prev_a = chunk[:, :10].float()
                        prev_m = (torch.arange(10, device=device)[None]
                                  < executed)
                        decision += 1
                        if term or trunc:
                            break
                    success = terminal_success(env)
                    record = {
                        "run_id": args.run_id, "task": task,
                        "seed": env_seed, "arm": arm,
                        "success": bool(success),
                        "q_valid": automaton.q_valid(),
                        "ordered_prefix": automaton.ordered_prefix(),
                        "p_valid": automaton.p_valid(),
                        "damage_unrecovered":
                            automaton.damage_unrecovered(),
                        "n_flips_down": sum(
                            1 for f in automaton.flips if f[2] == -1),
                        "first_unresolved": next(
                            (subgoals[i] for i, v in enumerate(
                                automaton.prev_valid) if not v), None),
                        "steps": t,
                        "terminated_early": bool(term or trunc),
                        "wall_seconds": round(
                            walltime.time() - started, 1),
                        "manifest_sha256": manifest_hash,
                    }
                finally:
                    env.close()
                np.savez_compressed(
                    ROOT / "traces"
                    / f"{args.run_id}_{task}_s{env_seed}_{arm}.npz",
                    actions_env=np.stack(actions_env),
                    noise_seeds=np.asarray(used_seeds))
                with records_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(f"[{task} seed={env_seed} arm={arm}] "
                      f"success={record['success']} "
                      f"q={record['q_valid']:.3f} steps={record['steps']}",
                      flush=True)
    print(f"-> {records_path}", flush=True)


if __name__ == "__main__":
    main()
