"""Probe-set collection for the world/task ladder (framework_design.md §3, §11.1).

Replays replay-valid expert demos (Stage-3 machinery), captures frames at a fixed
stride, and stores per frame:
  features : SigLIP pre-trunk | const-prompt last-layer | real-prompt last-layer
             (image tokens 2x2-pooled, fp16) + pooled real-prompt e_lang
  GT       : task-relevant object positions (sim), proprio q, task_id, phase t/T,
             remaining steps, episode success (from the demo's replay outcome)

Shards: one .pt per (task, demo) under <datasets>/probe_set/<suite>/.
Splits: demos are assigned to train/val/test by demo index (siblings stay together,
per the locked contract §1).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from lcwm.libero_paths import DATASETS_DIR, ensure_project_libero_config

ensure_project_libero_config()

from lcwm.chassis import Pi05Runner, make_task_env, make_task_suite  # noqa: E402
from lcwm.sampler import prefix_forward  # noqa: E402
from lcwm.snapshot import get_sim, reset_osc_controller  # noqa: E402
from lcwm.taps import CONSTANT_PROMPT, pool_image_tokens, pooled_lang  # noqa: E402

PROBE_DIR = DATASETS_DIR.parent / "probe_set"


def discover_object_bodies(env) -> dict[str, str]:
    """Map object name -> mujoco root body name for pose readout."""
    raw = env._env
    inner = getattr(raw, "env", raw)
    mapping = {}
    for name, obj in getattr(inner, "objects_dict", {}).items():
        root = getattr(obj, "root_body", None)
        if root is not None:
            mapping[name] = root
    if not mapping:  # fallback: robosuite objects list
        for obj in getattr(inner, "objects", []):
            mapping[obj.name] = obj.root_body
    return mapping


def body_positions(env, body_names: list[str]) -> np.ndarray:
    sim = get_sim(env)
    out = []
    for b in body_names:
        try:
            out.append(sim.data.get_body_xpos(b).copy())
        except Exception:
            bid = sim.model.body_name2id(b)
            out.append(sim.data.body_xpos[bid].copy())
    return np.stack(out)  # (n_obj, 3)


@torch.no_grad()
def frame_features(runner, obs, real_desc: str) -> dict[str, torch.Tensor]:
    const_b = runner._obs_to_policy_batch(obs, CONSTANT_PROMPT)
    real_b = runner._obs_to_policy_batch(obs, real_desc)

    pre_c = prefix_forward(runner.policy, const_b)
    pre_r = prefix_forward(runner.policy, real_b)

    valid = pre_c.pad_masks[0, : pre_c.n_img_tokens].bool()
    h_const = pre_c.hidden[0, : pre_c.n_img_tokens][valid].float()
    h_real = pre_r.hidden[0, : pre_r.n_img_tokens][valid].float()

    images, img_masks = runner.policy._preprocess_images(const_b)
    model = runner.policy.model
    sig = torch.cat(
        [model.paligemma_with_expert.embed_image(img)[0].float()
         for img, m in zip(images, img_masks) if bool(m[0])], dim=0)

    return {
        "siglip": pool_image_tokens(sig).half().cpu(),
        "const_ll": pool_image_tokens(h_const).half().cpu(),
        "real_ll": pool_image_tokens(h_real).half().cpu(),
        "e_lang": pooled_lang(pre_r).half().cpu(),
    }


def collect_task(
    runner,
    suite_name: str,
    task_id: int,
    demo_indices: list[int],
    stride: int = 5,
    out_dir: Path | None = None,
) -> list[Path]:
    suite = make_task_suite(suite_name)
    task = suite.get_task(task_id)
    real_desc = task.language
    h5_path = DATASETS_DIR / suite_name / f"{task.name}_demo.hdf5"
    out_dir = out_dir or (PROBE_DIR / suite_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_task_env(suite_name, task_id)
    env.reset()
    obj_bodies = discover_object_bodies(env)
    body_names = list(obj_bodies.values())

    written = []
    with h5py.File(h5_path, "r") as f:
        for di in demo_indices:
            key = f"demo_{di}"
            if key not in f["data"]:
                continue
            grp = f["data"][key]
            actions = np.asarray(grp["actions"])
            rec_states = np.asarray(grp["states"])
            T = len(actions)

            env._env.reset()
            sim = get_sim(env)
            sim.set_state_from_flattened(rec_states[0].copy())
            sim.forward()
            reset_osc_controller(env)

            rows = []
            raw = env._env
            success = False
            for t in range(T):
                if t % stride == 0:
                    obs = env._format_raw_obs(raw.env._get_observations())
                    feats = frame_features(runner, obs, real_desc)
                    rows.append({
                        **feats,
                        "obj_pos": torch.from_numpy(
                            body_positions(env, body_names)).float(),
                        "q": torch.from_numpy(np.concatenate([
                            obs["robot_state"]["eef"]["pos"],
                            obs["robot_state"]["eef"]["quat"],
                            obs["robot_state"]["gripper"]["qpos"],
                        ])).float(),
                        "t": t, "T": T,
                    })
                _o, _r, done, _i = raw.step(actions[t])
                success = success or bool(raw.check_success())
                if done:
                    break

            shard = {
                "suite": suite_name, "task_id": task_id, "demo": di,
                "task_language": real_desc, "object_names": list(obj_bodies),
                "body_names": body_names, "success": success,
                "stride": stride, "rows": rows,
            }
            p = out_dir / f"task{task_id}_demo{di}.pt"
            torch.save(shard, p)
            written.append(p)
            print(f"[probe] task{task_id} demo{di}: {len(rows)} frames, "
                  f"success={success} -> {p.name}", flush=True)
    env.close()
    return written


def valid_demo_indices(suite_name: str, task_id: int, k: int) -> list[int]:
    """First k replay-valid demo indices from the Stage-3 report."""
    rep_path = (Path(__file__).resolve().parent.parent
                / "results" / "replay_validity" / "replay_validity_report.json")
    rep = json.loads(rep_path.read_text())
    for tsk in rep["tasks"]:
        if tsk.get("task_id") == task_id and "demos" in tsk:
            idx = [int(d["demo_key"].split("_")[-1])
                   for d in tsk["demos"] if d["replay_valid"]]
            return idx[:k]
    raise RuntimeError(f"no replay report entry for task {task_id}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
    p.add_argument("--demos-per-task", type=int, default=10)
    p.add_argument("--stride", type=int, default=5)
    args = p.parse_args()

    runner = Pi05Runner(suite_name=args.suite)
    for tid in args.tasks:
        idx = valid_demo_indices(args.suite, tid, args.demos_per_task)
        collect_task(runner, args.suite, tid, idx, stride=args.stride)


if __name__ == "__main__":
    main()
