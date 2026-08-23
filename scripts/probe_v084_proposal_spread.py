#!/usr/bin/env python
"""Pre-spend probe for Action 2M.2: can effect strata be populated at all?

Action 2M.2 step 1 is "draw raw candidate chunks from stock PI0 exactly as
before", and step 3 selects a quota across effect strata.  Selection cannot
create spread that is absent from the pool, so this measures the RAW pool's
effect spread before a ~120k-step collection is committed to it.

Cost: `--sources N` source rollouts to tau only (250 steps each).  No branch is
executed, no bank is written, nothing trains.  Charged and reported as a
diagnostic.
"""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.snapshot import restore  # noqa: E402
from lcwm.v080r_panel import make_env_at  # noqa: E402
from lcwm.v081p_exec import SegmentLedger, run_source  # noqa: E402

TASK, DEADLINE, TAU, C = "chain1b_lr2", 250, 160, 10
PROBE_SEEDS = tuple(range(3800, 3816))
OUT = REPO / "results" / "v084_probe"


def proxies(chunk_env: np.ndarray) -> dict:
    """Action-only effect proxies over the executed first-c actions."""
    t = chunk_env[:, :3].sum(0)
    r = chunk_env[:, 3:6].sum(0)
    g = chunk_env[:, 6]
    tmag = float(np.linalg.norm(t))
    return {"tmag": tmag,
            "tdir": (t / tmag if tmag > 1e-9 else t * 0.0).tolist(),
            "rmag": float(np.linalg.norm(r)),
            "grip_closed": bool(g.mean() > 0),
            "grip_first_close": int(np.argmax(g > 0.5)) if bool((g > 0.5).any()) else C}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=int, default=6)
    ap.add_argument("--draws", type=int, default=64)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp
    out.mkdir(parents=True, exist_ok=True)
    seeds = PROBE_SEEDS[: args.sources]
    ledger = SegmentLedger(out / "segment_ledger.jsonl",
                           {"source": len(seeds) * DEADLINE}, action_root=OUT)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_env_at(TASK, DEADLINE)

    records = []
    for seed in seeds:
        s = run_source(runner, env, seed, ledger)
        a = s["anchor"]
        if a is None or not a.mask_ok or not s["failure_at_L"]:
            print(f"src {seed}: not an eligible anchor, skipped", flush=True)
            continue
        restore(env, a.snapshot)
        runner.reset()
        obs = env._format_raw_obs(env._env.env._get_observations())
        po = runner._obs_to_policy_batch(obs, env.task_description)
        pf = prefix_forward(runner.policy, po)
        chunks = sample_chunks(runner.policy, po, args.draws,
                               seed=abs(hash(("probe", seed))) % (2**31 - 1),
                               prefix=pf)[:, :C].detach().float().cpu()
        env_chunks = np.stack([runner.chunk_to_env(chunks[i]) for i in range(args.draws)])
        px = [proxies(c) for c in env_chunks]
        tm = np.array([p["tmag"] for p in px]); rm = np.array([p["rmag"] for p in px])
        gc = np.array([p["grip_closed"] for p in px])
        uniq = len({np.round(c, 4).tobytes() for c in env_chunks})
        rec = {"seed": seed, "anchor": a.anchor_id, "draws": args.draws,
               "unique_first_c": uniq,
               "tmag": {"min": float(tm.min()), "max": float(tm.max()),
                        "mean": float(tm.mean()), "std": float(tm.std()),
                        "cv": float(tm.std() / max(tm.mean(), 1e-9))},
               "rmag": {"min": float(rm.min()), "max": float(rm.max()),
                        "std": float(rm.std())},
               "grip_closed_fraction": float(gc.mean()),
               "grip_classes_present": int(len(set(gc.tolist()))),
               "max_pairwise_sum_gap": np.abs(
                   env_chunks.sum(1)[:, None] - env_chunks.sum(1)[None]).max(axis=(0, 1)).tolist()}
        records.append(rec)
        print(f"src {seed} {a.anchor_id}: unique={uniq}/{args.draws} "
              f"tmag cv={rec['tmag']['cv']:.4f} range={tm.max()-tm.min():.3f} "
              f"grip_closed={gc.mean():.2f} classes={rec['grip_classes_present']}",
              flush=True)

    summary = {"stamp": stamp, "task": TASK, "tau": TAU, "draws": args.draws,
               "eligible_anchors": len(records), "sources_run": len(seeds),
               "env_steps": ledger.total, "records": records}
    if records:
        summary["aggregate"] = {
            "median_tmag_cv": float(np.median([r["tmag"]["cv"] for r in records])),
            "median_tmag_range": float(np.median(
                [r["tmag"]["max"] - r["tmag"]["min"] for r in records])),
            "anchors_with_both_grip_classes": int(sum(
                r["grip_classes_present"] > 1 for r in records)),
            "median_unique_fraction": float(np.median(
                [r["unique_first_c"] / r["draws"] for r in records]))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary.get("aggregate", {}), indent=2))
    print(f"env steps {ledger.total} | artifact {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
