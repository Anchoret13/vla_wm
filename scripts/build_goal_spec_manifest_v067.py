#!/usr/bin/env python
"""V6.7.1 — extend the frozen GoalSpec manifest with explicit terminal
predicates (distinct from event milestones such as pick_up).

The frozen v1 manifest is reused by hash reference, never edited. For
every GoalSpec (canonical + scene-valid distinct goals) this derives the
terminal predicate list from its non-pick_up ordered subgoals, and for
each task's CANONICAL GoalSpec asserts set-equality against the
environment's BDDL `goal_state` (the only place the BDDL evaluator is
authoritative). Output: goal_spec_manifest_v067.json (run_schema=v067).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
V1_PATH = RESULTS / "goal_spec_manifest.json"
OUT_PATH = RESULTS / "goal_spec_manifest_v067.json"


def main() -> None:
    from lcwm.goal_semantics import (bddl_goal_predicates,
                                     goal_terminal_success,
                                     terminal_predicate_hash,
                                     terminal_predicates)
    from lcwm.loho_public import make_public_env
    from lcwm.task_automaton import terminal_success
    from lcwm.v067_lineage import RUN_SCHEMA, sha256_file

    v1 = json.loads(V1_PATH.read_text())
    out = {"schema": "goal_spec_manifest_v067", "run_schema": RUN_SCHEMA,
           "reused_v1_manifest_sha256": sha256_file(V1_PATH),
           "note": ("terminal predicates derived from non-pick_up ordered "
                    "subgoals; canonical set-equality vs BDDL goal_state "
                    "asserted per task; pick_up is an event milestone and "
                    "never terminal"),
           "tasks": {}}

    for task_name, spec in v1["tasks"].items():
        specs = [dict(spec["canonical"])] + [dict(g) for g in
                                             spec["distinct_goals"]]
        env = make_public_env(task_name, 300)
        try:
            env.reset(seed=0)
            bddl = {tuple(str(x).lower() for x in p)
                    for p in bddl_goal_predicates(env)}
            entry = {"goal_specs": {}}
            for gs in specs:
                preds = terminal_predicates(gs["ordered_subgoals"])
                gid = gs["goal_spec_id"]
                entry["goal_specs"][gid] = {
                    "language": gs["language"],
                    "prompt_sha256": hashlib.sha256(
                        gs["language"].encode()).hexdigest()[:16],
                    "ordered_subgoals": gs["ordered_subgoals"],
                    "terminal_predicates": [list(p) for p in preds],
                    "terminal_predicate_hash":
                        terminal_predicate_hash(preds),
                }
            canon_id = spec["canonical"]["goal_spec_id"]
            canon_preds = {
                tuple(str(x).lower() for x in p) for p in
                entry["goal_specs"][canon_id]["terminal_predicates"]}
            assert canon_preds == bddl, (
                f"{task_name}: derived canonical terminal predicates != "
                f"BDDL goal_state\n derived={sorted(canon_preds)}\n "
                f"bddl={sorted(bddl)}")
            # functional agreement at the reset state as well
            derived_now = goal_terminal_success(
                env, [tuple(p) for p in
                      entry["goal_specs"][canon_id]["terminal_predicates"]])
            assert derived_now == terminal_success(env), task_name
            entry["canonical_goal_spec_id"] = canon_id
            entry["bddl_goal_state_lower"] = sorted(
                [list(p) for p in bddl])
            out["tasks"][task_name] = entry
            print(f"[ok] {task_name}: canonical == BDDL "
                  f"({len(bddl)} predicates), "
                  f"{len(specs)} goal specs", flush=True)
        finally:
            env.close()

    payload = json.dumps(out, indent=2, sort_keys=True)
    out["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
