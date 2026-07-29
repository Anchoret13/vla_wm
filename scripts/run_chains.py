#!/usr/bin/env python
"""Chained-exam runner (minimal scope: numbers then stop, per 2026-07-22).

3 tasks × {full, decomp} × 5 episodes, seeds 1000+. Frozen pi0.5, zero
finetuning. Output: results/chains/chain_exam.json + printed table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from dataclasses import asdict  # noqa: E402

from lcwm.chassis import Pi05Runner  # noqa: E402
from lcwm.loho import CHAINS, make_chain_env, run_chain_episode  # noqa: E402

EPISODES = 5


def main() -> None:
    runner = Pi05Runner(suite_name="libero_10")
    out_dir = REPO_ROOT / "results" / "chains"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    jl = out_dir / "chain_exam_v2.jsonl"   # v2: no-auto-reset ChainEnv   # incremental: a crash never loses data
    jl.write_text("")
    for name in CHAINS:
        env = make_chain_env(name)
        for condition in ("full", "decomp"):
            for ep in range(EPISODES):
                res = run_chain_episode(runner, env, condition, seed=1000 + ep)
                results.append(asdict(res))
                with jl.open("a") as f:
                    f.write(json.dumps(asdict(res)) + "\n")
                print(f"[{name}/{condition}] ep{ep} seed{1000+ep}: "
                      f"success={res.success} q={res.q_score:.2f} "
                      f"steps={res.steps} instr_used={len(res.instructions_used)}",
                      flush=True)
        env.close()

    (out_dir / "chain_exam_v2.json").write_text(json.dumps(results, indent=1))
    print(f"\n{'task':12s} {'cond':7s} {'SR':>5s} {'Q':>6s}")
    for name in CHAINS:
        for condition in ("full", "decomp"):
            rs = [r for r in results
                  if r["task"] == name and r["condition"] == condition]
            sr = sum(r["success"] for r in rs) / len(rs)
            q = sum(r["q_score"] for r in rs) / len(rs)
            print(f"{name:12s} {condition:7s} {sr:5.0%} {q:6.2f}")
    print(f"-> {out_dir / 'chain_exam_v2.json'}")


if __name__ == "__main__":
    main()
