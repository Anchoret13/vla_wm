#!/usr/bin/env python
"""H7.1 exit check — one-seed stock π0.5 smoke on the five reconstructed
public tasks. Records per task: reset validity, subgoal/predicate timeline,
terminal capture, public Q. Appends to evaluator_smoke.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

SMOKE_SEED = 1300
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}


def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.loho_public import (
        MANIFEST_PATH,
        load_public_tasks,
        make_public_env,
        run_public_episode,
    )

    tasks = load_public_tasks("reconstructed")
    assert len(tasks) == 5, f"expected 5 reconstructed tasks, got {len(tasks)}"
    runner = Pi05Runner(suite_name="libero_10")
    out = MANIFEST_PATH.parent / "evaluator_smoke.jsonl"
    for name, spec in sorted(tasks.items()):
        env = make_public_env(name, EPISODE_LENGTH[name])
        try:
            result = run_public_episode(
                runner, env, spec["ordered_subgoals"], seed=SMOKE_SEED
            )
        finally:
            env.close()
        record = {"task": name, "arm": "stock", **result}
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"[smoke] {name}: Q_public={result['q_public']:.3f} "
            f"success={result['success']} steps={result['steps']} "
            f"first_unresolved={result['first_unresolved_subgoal']} "
            f"completed={list(result['subgoal_completion_steps'])}",
            flush=True,
        )
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
