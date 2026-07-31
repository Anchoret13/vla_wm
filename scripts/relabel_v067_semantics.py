#!/usr/bin/env python
"""V6.7.1 — regenerate full-path semantic labels at ACTION resolution
(replaces iteration-1 `labels/` [v06_labels_v1], which a v067 job
refuses to load).

Per source: deterministic env-action replay from reset (recorded-anchor
re-staging), evaluating EVERY registered GoalSpec's automaton after
every environment action. Object positions asserted against the phase-A
record at every decision. Per decision × goal this stores automaton
state before/after, the decision-window flip log, ordered prefix, and
GoalSpec-specific terminal-predicate status (any-point within the
window + at the boundary).

Output: v06_effect_crossed/semantic_relabels_v067/<source>.pt
(run_schema=v067). The reused fp16 h feature sidecars are referenced by
hash in manifest_v067.json and are NOT regenerated here.
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

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
REREACH_ATOL = 2e-3


def main() -> None:
    from lcwm.goal_semantics import (env_eval_fn, goal_terminal_success)
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.task_automaton import GoalAutomaton, fork_env_state
    from lcwm.v067_lineage import RUN_SCHEMA, assert_v067_payload
    from scripts.collect_loho_v06 import stage_support

    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    assert_v067_payload(goal_manifest, "goal_spec_manifest_v067.json",
                        "goal_manifest")
    out_dir = DATA / "semantic_relabels_v067"
    out_dir.mkdir(exist_ok=True)

    for path in sorted((DATA / "sources").glob("*.pt")):
        out_path = out_dir / path.name
        if out_path.exists():
            print(f"[skip] {path.stem}", flush=True)
            continue
        source = torch.load(path, weights_only=False)
        assert source["schema"] == "v06_source_v1"
        task_name = source["task"]
        entry = goal_manifest["tasks"][task_name]
        canon_id = entry["canonical_goal_spec_id"]
        goal_specs = entry["goal_specs"]
        goal_ids = [canon_id] + [g for g in goal_specs if g != canon_id]
        term_preds = {gid: [tuple(p) for p in
                            goal_specs[gid]["terminal_predicates"]]
                      for gid in goal_ids}

        env = make_public_env(task_name, EPISODE_LENGTH[task_name])
        try:
            env.reset(seed=source["seed"])
            eval_fn = env_eval_fn(env)
            automata = {}
            for gid in goal_ids:
                a = GoalAutomaton(goal_specs[gid]["ordered_subgoals"])
                a.start(env)
                automata[gid] = a
            if source["provenance"] == "staged":
                stage_support(env, automata[canon_id], task_name,
                              recorded=source["staging_info"])
            for a in automata.values():
                a.evaluate(env, 0)
            bodies = list(automata[canon_id].bodies.values())

            per_decision = []
            t = 0
            for row in source["rows"]:
                dec = {"decision": row["decision"],
                       "t_start": row["t_start"], "goals": {}}
                fork_before = {gid: fork_env_state(a)
                               for gid, a in automata.items()}
                flips_before = {gid: len(a.flips)
                                for gid, a in automata.items()}
                term_any = {gid: goal_terminal_success(
                    env, term_preds[gid], eval_fn) for gid in goal_ids}
                for a_env in row["actions_env"]:
                    env.step(a_env)
                    t += 1
                    for gid, a in automata.items():
                        a.evaluate(env, t)
                        if not term_any[gid]:
                            term_any[gid] = goal_terminal_success(
                                env, term_preds[gid], eval_fn)
                err = float(np.abs(
                    body_positions(env, bodies)
                    - row["obj_after"]).max())
                assert err < REREACH_ATOL, (
                    f"{source['source_id']} d={row['decision']}: "
                    f"relabel replay diverges {err:.2e}")
                for gid, a in automata.items():
                    window = a.flips[flips_before[gid]:]
                    before = fork_before[gid]
                    dec["goals"][gid] = {
                        "state_before": before,
                        "state_after": fork_env_state(a),
                        "valid_before": list(before["prev_valid"]),
                        "valid_after": list(a.prev_valid),
                        "events_before": sorted(
                            before["events_achieved"]),
                        "events_after": sorted(a.events_achieved),
                        "flips_01": [f for f in window if f[2] == 1],
                        "flips_10": [f for f in window if f[2] == -1],
                        "ordered_prefix_before": next(
                            (i for i, v
                             in enumerate(before["prev_valid"])
                             if not v), len(before["prev_valid"])),
                        "ordered_prefix_after": a.ordered_prefix(),
                        "terminal_any_in_window": bool(term_any[gid]),
                        "terminal_at_boundary": goal_terminal_success(
                            env, term_preds[gid], eval_fn),
                        "terminal_predicate_hash":
                            goal_specs[gid]["terminal_predicate_hash"],
                    }
                per_decision.append(dec)
            torch.save({
                "schema": "v067_semantic_labels_v1",
                "run_schema": RUN_SCHEMA,
                "source_id": source["source_id"],
                "task": task_name, "split": source["split"],
                "goals": goal_ids,
                "canonical_goal_spec_id": canon_id,
                "goal_manifest_sha256": goal_manifest["manifest_sha256"],
                "per_decision": per_decision,
            }, out_path)
            print(f"[relabel] {source['source_id']}: "
                  f"{len(per_decision)} decisions x {len(goal_ids)} "
                  "goals (per-action)", flush=True)
        finally:
            env.close()
    print("v067 semantic relabel complete", flush=True)


if __name__ == "__main__":
    main()
