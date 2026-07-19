"""Stage-3 replay-validity protocol (plan §4).

Per suite: for each recorded demo, reset env to the demo's init state, replay the
recorded action sequence open-loop, and compare the resulting success flag (and
proprio trace divergence) against the demo. Literature reference range: 26-28 / 30
demos valid. Emits replay_validity_report.json; downstream, only snapshots from
valid replays may serve as counterfactual-GT anchors.

Demo source: original-format LIBERO hdf5 (keys: data/demo_k/{actions,states,obs/...}).
Dataset files live under lcwm.libero_paths.DATASETS_DIR (populated by the
libero_90 acquisition lane); this module only needs `actions` + init state.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np

from lcwm.libero_paths import DATASETS_DIR, ensure_project_libero_config

ensure_project_libero_config()

from lcwm.chassis import make_task_env, make_task_suite  # noqa: E402
from lcwm.snapshot import get_sim, reset_osc_controller  # noqa: E402


@dataclass
class DemoReplayResult:
    demo_key: str
    success: bool
    n_steps: int
    final_qpos_l2: float          # ‖replayed − recorded‖ at the last comparable step
    max_qpos_l2: float
    replay_valid: bool


@dataclass
class TaskReplayReport:
    suite: str
    task_id: int
    task_name: str
    hdf5: str
    n_demos: int
    n_valid: int
    demos: list[DemoReplayResult] = field(default_factory=list)


def _demo_file(suite_name: str, task) -> Path:
    """Original LIBERO naming: <datasets>/<suite>/<task.name>_demo.hdf5."""
    return DATASETS_DIR / suite_name / f"{task.name}_demo.hdf5"


def replay_task_demos(
    suite_name: str,
    task_id: int,
    qpos_tol: float = 5e-2,
    max_demos: int | None = None,
) -> TaskReplayReport:
    suite = make_task_suite(suite_name)
    task = suite.get_task(task_id)
    h5_path = _demo_file(suite_name, task)
    if not h5_path.exists():
        raise FileNotFoundError(
            f"demo hdf5 not found: {h5_path} — run the dataset acquisition lane first"
        )

    env = make_task_env(suite_name, task_id)
    env.reset()
    sim = get_sim(env)
    report = TaskReplayReport(
        suite=suite_name, task_id=task_id, task_name=task.name,
        hdf5=str(h5_path), n_demos=0, n_valid=0,
    )

    with h5py.File(h5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]
        for key in demo_keys:
            grp = f["data"][key]
            actions = np.asarray(grp["actions"])            # (T, 7)
            rec_states = np.asarray(grp["states"])          # (T, D) mujoco flat states
            # init from the demo's first recorded sim state
            sim.set_state_from_flattened(rec_states[0].copy())
            sim.forward()
            reset_osc_controller(env)

            success, dists = False, []
            raw = env._env
            for t in range(len(actions)):
                _obs, _r, done, _info = raw.step(actions[t])
                cur = np.asarray(sim.get_state().flatten())
                m = min(len(cur), rec_states.shape[1])
                dists.append(float(np.linalg.norm(cur[:m] - rec_states[t][:m])
                                   / np.sqrt(m)))
                success = success or bool(raw.check_success())
                if done:
                    break

            res = DemoReplayResult(
                demo_key=key,
                success=success,
                n_steps=len(dists),
                final_qpos_l2=dists[-1] if dists else float("inf"),
                max_qpos_l2=max(dists) if dists else float("inf"),
                replay_valid=bool(success),   # v0 criterion: replay reaches success
            )
            report.demos.append(res)
            report.n_demos += 1
            report.n_valid += int(res.replay_valid)
    env.close()
    return report


def write_suite_report(suite_name: str, out_dir: str | Path,
                       task_ids: list[int] | None = None,
                       max_demos: int | None = None) -> Path:
    suite = make_task_suite(suite_name)
    ids = task_ids if task_ids is not None else list(range(suite.n_tasks))
    reports = []
    for tid in ids:
        try:
            reports.append(asdict(replay_task_demos(suite_name, tid, max_demos=max_demos)))
        except FileNotFoundError as e:
            reports.append({"suite": suite_name, "task_id": tid, "error": str(e)})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "replay_validity_report.json"
    path.write_text(json.dumps({"suite": suite_name, "tasks": reports}, indent=2))
    return path
