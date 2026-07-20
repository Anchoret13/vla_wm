#!/usr/bin/env python
"""Layout probe set — properly-powered decision-time layout-reading test.

PRE-REGISTERED (2026-07-19.md): for each libero_10 task, reset to ALL 50 init
states (static, settled, no policy rollout), tap {siglip, const_ll, real_ll}
agentview tokens + GT object positions + proprio q. Probe: LocProbe trained on
init states 0–39, tested on 40–49 (unseen layouts). Eval dims = init-varying
dims on TRAIN (std > 5mm). PASS rule per stream: test pooled R² > 0.3 AND
margin over the q-only ridge control > 0.3.

Failure of all streams ⇒ frozen-feature layout reading needs auxiliary
supervision (Candidate-D element) or must be learned implicitly by WM dynamics —
to be recorded in framework_design.md either way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.chassis import Pi05Runner, make_task_env, make_task_suite  # noqa: E402
from lcwm.probe_data import (body_positions,  # noqa: E402
                             discover_object_bodies, frame_features)
from lcwm.snapshot import reset_osc_controller  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from probe_ladder import ridge_fit  # noqa: E402
from probe_world_controls import pooled_r2_err, train_locprobe  # noqa: E402

from lerobot.envs.libero import get_libero_dummy_action  # noqa: E402

LAYOUT_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets/probe_set/layout_libero_10")
STREAMS = ["siglip", "const_ll", "real_ll"]
SEEDS = (0, 1, 2)
SETTLE = 10


def collect_task(runner, suite_name: str, tid: int) -> Path:
    out = LAYOUT_DIR / f"task{tid}.pt"
    if out.exists():
        return out
    suite = make_task_suite(suite_name)
    task = suite.get_task(tid)
    init_states = suite.get_task_init_states(tid)
    env = make_task_env(suite_name, tid)
    env.reset()
    obj_bodies = discover_object_bodies(env)
    body_names = list(obj_bodies.values())
    raw = env._env

    rows = []
    for k in range(len(init_states)):
        raw.reset()
        raw.set_init_state(init_states[k])
        reset_osc_controller(env)
        for _ in range(SETTLE):
            raw.step(get_libero_dummy_action())
        obs = env._format_raw_obs(raw.env._get_observations())
        feats = frame_features(runner, obs, task.language)
        rows.append({
            **feats,
            "obj_pos": torch.from_numpy(body_positions(env, body_names)).float(),
            "q": torch.from_numpy(np.concatenate([
                obs["robot_state"]["eef"]["pos"],
                obs["robot_state"]["eef"]["quat"],
                obs["robot_state"]["gripper"]["qpos"],
            ])).float(),
            "init_id": k,
        })
    env.close()
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"task_id": tid, "object_names": list(obj_bodies),
                "language": task.language, "rows": rows}, out)
    print(f"[layout] task{tid}: {len(rows)} init states -> {out.name}", flush=True)
    return out


def eval_task(shard, dev="cuda"):
    rows = shard["rows"]
    tr_idx = [r["init_id"] < 40 for r in rows]
    te_idx = [r["init_id"] >= 40 for r in rows]
    res = {}
    ytr = torch.stack([r["obj_pos"] for r, m in zip(rows, tr_idx) if m]).to(dev)
    yte = torch.stack([r["obj_pos"] for r, m in zip(rows, te_idx) if m]).to(dev)
    qtr = torch.stack([r["q"] for r, m in zip(rows, tr_idx) if m]).to(dev)
    qte = torch.stack([r["q"] for r, m in zip(rows, te_idx) if m]).to(dev)
    varying = ytr.std(0).reshape(-1) > 0.005  # init-varying dims (train)
    if not bool(varying.any()):
        return None

    w, xm, ym = ridge_fit(qtr, ytr.reshape(len(ytr), -1), 1e0)
    predq = ((qte - xm) @ w + ym).reshape(yte.shape)
    res["q_only"] = pooled_r2_err(predq, yte, varying)

    for stream in STREAMS:
        ttr = torch.stack([r[stream][:64].float()
                           for r, m in zip(rows, tr_idx) if m]).to(dev)
        tte = torch.stack([r[stream][:64].float()
                           for r, m in zip(rows, te_idx) if m]).to(dev)
        r2s, errs = [], []
        for s in SEEDS:
            pred = train_locprobe(ttr, ytr, s)(tte)
            r2, err = pooled_r2_err(pred, yte, varying)
            r2s.append(r2)
            errs.append(err)
        res[stream] = (float(np.mean(r2s)), float(np.mean(errs)))
    res["n_varying_dims"] = int(varying.sum())
    return res


def main() -> None:
    runner = Pi05Runner(suite_name="libero_10")
    report = {}
    for tid in range(10):
        collect_task(runner, "libero_10", tid)
    del runner
    torch.cuda.empty_cache()

    agg = {s: [] for s in STREAMS + ["q_only"]}
    for tid in range(10):
        shard = torch.load(LAYOUT_DIR / f"task{tid}.pt", weights_only=False)
        r = eval_task(shard)
        if r is None:
            continue
        report[tid] = {k: v for k, v in r.items()}
        for s in STREAMS + ["q_only"]:
            agg[s].append(r[s][0])
        print(f"task{tid}: " + "  ".join(
            f"{s} {r[s][0]:.3f}[{r[s][1]:.1f}cm]" for s in STREAMS + ["q_only"]))

    print("\n=== LAYOUT READING (40 train / 10 test init states) ===")
    for s in STREAMS + ["q_only"]:
        print(f"{s:10s} mean R2 {np.mean(agg[s]):.3f}")
    outp = REPO_ROOT / "results" / "layout_probe.json"
    outp.write_text(json.dumps({str(k): v for k, v in report.items()}, indent=2))
    print("pass rule: stream R2 > 0.3 AND (stream - q_only) > 0.3")
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
