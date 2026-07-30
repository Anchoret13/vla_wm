#!/usr/bin/env python
"""H7.5 — paired public LIBERO-LoHo development matrix evaluator.

Contract (mirrors eval_long_horizon_paired.py, adapted to the public
five-task protocol):
- noise_seed(task, env_seed, decision) = 20e6 + public_task_index*2e6 +
  env_seed*1e3 + decision; the stream is shared exactly across arms;
- every arm samples through the SAME seeded LC code path
  (sample_chunks_lc); stock = bias ≡ 0 (exact parity by construction);
- full canonical prompt only; H7.1 subgoal protocol (SubgoalTracker every
  10 steps; SR = terminal goal-atom conjunction; Q = completed/registered);
- per-run immutable manifest; one record per (run_id, task, seed, arm);
  --resume verifies the manifest hash; hash-permutation arm interleaving;
- per-episode action traces; later-invalidation = completed subgoal whose
  predicate is false at the terminal state.

Arms:
- stock                     zero bias
- v05_reset_wm / v05_reset_random / v05_recurrent_wm /
  v05_recurrent_random     frozen v0.5 WM + trained interface
                            (state mode from the adapter bundle)
- v04_cur_wm               frozen v0.4 current-mode legacy reference
"""

from __future__ import annotations

import argparse
import copy
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
ROOT = REPO_ROOT / "results" / "libero_loho_public_v1" / "eval_matrix"
WM_CKPT = (REPO_ROOT / "results" / "libero_loho_public_v1" / "v05_gated_wm_r1"
           / "checkpoint_final.pt")
V04_ADAPTER = (REPO_ROOT / "results" / "lc_flow_v04_e2e_v1_cur_wm"
               / "eval_adapter")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_seed(task: str, env_seed: int, decision: int) -> int:
    return (NOISE_BASE + TASK_ORDER.index(task) * 2_000_000
            + env_seed * 1_000 + decision)


def arm_order(task: str, env_seed: int, arms: list[str]) -> list[str]:
    return sorted(arms, key=lambda arm: hashlib.sha256(
        f"{task}|{env_seed}|{arm}".encode()).hexdigest())


def obs_q(obs) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([
        obs["robot_state"]["eef"]["pos"],
        obs["robot_state"]["eef"]["quat"],
        obs["robot_state"]["gripper"]["qpos"],
    ])).float()


class V05Deploy:
    """Live c/w/g computation + gated bias for one v0.5 arm."""

    def __init__(self, runner, model, state_mode: str, device):
        from lcwm.taps import pool_image_tokens
        self.runner, self.model, self.state_mode = runner, model, state_mode
        self.device = device
        self.pool_image_tokens = pool_image_tokens
        self.zero_a = torch.zeros(1, 10, 7, device=device)
        self.zero_m = torch.zeros(1, 10, dtype=torch.bool, device=device)
        self.reset()

    def reset(self):
        self.w = self.g = self.a_prev = self.m_prev = None

    @torch.no_grad()
    def bias(self, batch, prefix, obs):
        model, device = self.model, self.device
        images, img_masks = self.runner.policy._preprocess_images(batch)
        pi = self.runner.policy.model
        sig = torch.cat(
            [pi.paligemma_with_expert.embed_image(img)[0].float()
             for img, m in zip(images, img_masks) if bool(m[0])], dim=0)
        h_early = self.pool_image_tokens(sig)[None].float()
        h_late = prefix.hidden.float()
        mask = prefix.pad_masks.bool()
        q = obs_q(obs)[None].to(device)
        w0, g0 = model.initial(1, device)
        if self.state_mode == "reset" or self.w is None:
            w = model.step_physical(w0, self.zero_a, h_early, q,
                                    action_mask=self.zero_m)
            g = model.step_task(g0, w, self.zero_a, h_late, mask,
                                action_mask=self.zero_m)
        else:
            w = model.step_physical(self.w, self.a_prev, h_early, q,
                                    action_mask=self.m_prev)
            g = model.step_task(self.g, w, self.a_prev, h_late, mask,
                                action_mask=self.m_prev)
        c = model.current(h_early, h_late, mask, q)
        if self.state_mode == "recurrent":
            self.w, self.g = w, g
        return model.policy_bias(c, w, g)

    def after_execute(self, chunk10, executed: int):
        if self.state_mode == "recurrent":
            self.a_prev = chunk10.float()
            self.m_prev = (torch.arange(10, device=self.device)[None]
                           < executed)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", default=TASK_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--panel", default="development")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import load_lc_adapter, sample_chunks_lc
    from lcwm.loho_public import (SubgoalTracker, _problem_env,
                                  load_public_tasks, make_public_env)
    from lcwm.sampler import prefix_forward
    from lcwm.v05_model import V05State

    panel_dir = ROOT / args.panel
    panel_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "traces").mkdir(exist_ok=True)
    records_path = panel_dir / "records.jsonl"
    tasks_spec = load_public_tasks()

    adapter_files = {}
    for arm in args.arms:
        if arm.startswith("v05_"):
            adapter_files[arm] = (
                REPO_ROOT / "results" / "libero_loho_public_v1"
                / "adapters" / arm / "adapter.pt")
        elif arm == "v04_cur_wm":
            adapter_files[arm] = V04_ADAPTER / "lc_flow.safetensors"

    manifest = {
        "schema": "loho_public_eval_v1",
        "run_id": args.run_id,
        "noise_contract": ("20e6 + public_task_index*2e6 + env_seed*1e3 "
                           "+ decision; shared across arms; all arms "
                           "sample via sample_chunks_lc (stock bias=0)"),
        "prompt_mode": "canonical_full",
        "arms": sorted(args.arms),
        "adapter_hashes": {a: sha256_file(p)
                           for a, p in adapter_files.items()},
        "wm_checkpoint_sha256": (sha256_file(WM_CKPT)
                                 if any(a.startswith("v05_")
                                        for a in args.arms) else None),
        "tasks": args.tasks,
        "seeds": args.seeds,
        "code_files": {f: sha256_file(REPO_ROOT / f) for f in (
            "scripts/eval_loho_public_paired.py", "lcwm/loho_public.py",
            "lcwm/v05_model.py", "lcwm/lc_flow.py")},
        "bddl_hashes": {t: sha256_file(
            REPO_ROOT / "bddl" / "libero_loho_public_v1" / f"{t}.bddl")
            for t in args.tasks},
    }
    payload = json.dumps(manifest, sort_keys=True)
    manifest_hash = hashlib.sha256(payload.encode()).hexdigest()
    manifest["manifest_sha256"] = manifest_hash
    manifest_file = ROOT / f"run_manifest_{args.run_id}.json"
    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text())
        assert existing["manifest_sha256"] == manifest_hash, (
            "manifest hash mismatch on resume — aborting")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2))
    done_keys = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done_keys.add((r["run_id"], r["task"], r["seed"], r["arm"]))
        if not args.resume:
            assert not any(k[0] == args.run_id for k in done_keys), (
                "records exist for this run_id; use --resume")

    runner = Pi05Runner(suite_name="libero_10")
    zero_bias = torch.zeros(1, 1024, device=device)

    v05_model = None
    if any(a.startswith("v05_") for a in args.arms):
        v05_model = V05State().to(device)
        v05_model.load_state_dict(
            torch.load(WM_CKPT, weights_only=False)["model"])
        v05_model.eval()
    v04_lc = None
    if "v04_cur_wm" in args.arms:
        v04_lc, _ = load_lc_adapter(V04_ADAPTER, device=str(device))

    def make_arm_state(arm: str):
        if arm.startswith("v05_"):
            bundle = torch.load(adapter_files[arm], weights_only=False)
            assert bundle["wm_checkpoint_sha256"] == \
                manifest["wm_checkpoint_sha256"]
            model = copy.deepcopy(v05_model)
            model.load_state_dict(
                {**model.state_dict(), **bundle["interface"]})
            return V05Deploy(runner, model, bundle["state_mode"], device)
        return None

    ledger_path = panel_dir / "ledger.jsonl"
    for task in args.tasks:
        subgoals = tasks_spec[task]["ordered_subgoals"]
        for env_seed in args.seeds:
            for arm in arm_order(task, env_seed, list(args.arms)):
                key = (args.run_id, task, env_seed, arm)
                if key in done_keys:
                    continue
                started = walltime.time()
                with ledger_path.open("a") as ledger:
                    ledger.write(json.dumps(
                        {"key": key, "status": "running"}) + "\n")
                deploy = make_arm_state(arm)
                env = make_public_env(task, EPISODE_LENGTH[task])
                try:
                    runner.reset()
                    obs, _ = env.reset(seed=env_seed)
                    tracker = SubgoalTracker(list(subgoals))
                    tracker.start(env)
                    instruction = env.task_description
                    inner = _problem_env(env)
                    goal_atoms = [list(a) for a in
                                  inner.parsed_problem["goal_state"]]
                    actions_env, used_seeds = [], []
                    t, decision = 0, 0
                    term = trunc = False
                    while t < env.episode_length:
                        batch = runner._obs_to_policy_batch(
                            obs, instruction)
                        prefix = prefix_forward(runner.policy, batch)
                        if deploy is not None:
                            bias = deploy.bias(batch, prefix, obs)
                        elif arm == "v04_cur_wm":
                            z = v04_lc.posterior(
                                prefix.hidden, prefix.pad_masks)
                            bias = v04_lc.adarms_bias(z)
                        else:
                            bias = zero_bias
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
                        tracker.update(env, t)
                        if deploy is not None:
                            deploy.after_execute(chunk[:, :10], executed)
                        decision += 1
                        if term or trunc:
                            break
                    success = all(bool(inner._eval_predicate(a))
                                  for a in goal_atoms)
                    invalidated = []
                    for i in sorted(tracker.completed):
                        parts = subgoals[i].split()
                        if parts[0] == "place":
                            pred = ("on" if parts[2].endswith("top_side")
                                    else "in")
                            ok = bool(inner._eval_predicate(
                                [pred, parts[1], parts[2]]))
                        elif parts[0] in ("open", "close"):
                            ok = bool(inner._eval_predicate(
                                [parts[0], parts[1]]))
                        else:
                            ok = True  # pick_up: geometric, not re-checked
                        if not ok:
                            invalidated.append(subgoals[i])
                    record = {
                        "run_id": args.run_id, "task": task,
                        "seed": env_seed, "arm": arm,
                        "prompt_mode": "canonical_full",
                        "success": bool(success),
                        "q_public": tracker.q_score(),
                        "ordered_prefix": tracker.ordered_prefix(),
                        "subgoal_completion_steps": {
                            subgoals[i]: s for i, s in
                            sorted(tracker.completed.items())},
                        "invalidated_subgoals": invalidated,
                        "first_unresolved_subgoal": next(
                            (subgoals[i] for i in range(len(subgoals))
                             if i not in tracker.completed), None),
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
                with ledger_path.open("a") as ledger:
                    ledger.write(json.dumps(
                        {"key": key, "status": "complete",
                         "success": record["success"],
                         "q": record["q_public"]}) + "\n")
                print(f"[{task} seed={env_seed} arm={arm}] "
                      f"success={record['success']} "
                      f"q={record['q_public']:.3f} "
                      f"steps={record['steps']}", flush=True)
    print(f"-> {records_path}", flush=True)


if __name__ == "__main__":
    main()
