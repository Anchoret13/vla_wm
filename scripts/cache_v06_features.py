#!/usr/bin/env python
"""V6.2/V6.3 — crossed feature + label cache for v0.6 training.

Per source, two passes:
1. ENV pass (no policy): deterministic env-action replay from reset
   (recorded-anchor re-staging for staged sources), evaluating EVERY
   registered GoalSpec's automaton at each decision boundary. Object
   positions asserted against the phase-A record. Output: per-decision
   per-goal current labels (valid bits, event bits, ordered prefix) and
   canonical reward deltas.
2. GPU pass: for each decision's stored raw observation × each registered
   text variant (canonical, both paraphrases, every distinct GoalSpec's
   canonical language): recompute the full π0.5 prefix (hash-addressed by
   (source, decision, prompt sha)); store h (fp16) + pad mask.

Crossed rows thereby reference ONE stored physical transition and
separate semantic labels; prompt-bound features are never copied across
texts (each variant is its own forward).

Output:
  v06_effect_crossed/labels/<source>.pt
  v06_effect_crossed/features/<source>__<variant_id>.pt
"""

from __future__ import annotations

import hashlib
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
GOAL_SPECS = json.loads(
    (REPO_ROOT / "results" / "libero_loho_public_v1"
     / "goal_spec_manifest.json").read_text())
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
REREACH_ATOL = 2e-3


def variants_for(task_name: str) -> list[dict]:
    spec = GOAL_SPECS["tasks"][task_name]
    out = [{"variant_id": "canonical",
            "goal_spec_id": spec["canonical"]["goal_spec_id"],
            "language": spec["canonical"]["language"],
            "kind": "canonical"}]
    for i, p in enumerate(spec["paraphrases"]):
        out.append({"variant_id": f"paraphrase_{i}",
                    "goal_spec_id": spec["canonical"]["goal_spec_id"],
                    "language": p, "kind": "paraphrase"})
    for g in spec["distinct_goals"]:
        out.append({"variant_id": f"distinct_{g['goal_spec_id']}",
                    "goal_spec_id": g["goal_spec_id"],
                    "language": g["language"], "kind": "distinct"})
    return out


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward
    from lcwm.task_automaton import GoalAutomaton, fork_env_state
    from scripts.collect_loho_v06 import stage_support

    (DATA / "labels").mkdir(exist_ok=True)
    (DATA / "features").mkdir(exist_ok=True)
    runner = None  # lazy: env pass first for all sources

    sources = sorted((DATA / "sources").glob("*.pt"))

    # ---- pass 1: env labels --------------------------------------------
    for path in sources:
        source = torch.load(path, weights_only=False)
        out_path = DATA / "labels" / path.name
        if out_path.exists():
            continue
        task_name = source["task"]
        spec = GOAL_SPECS["tasks"][task_name]
        goals = [{"goal_spec_id": spec["canonical"]["goal_spec_id"],
                  "ordered_subgoals":
                      spec["canonical"]["ordered_subgoals"]}]
        goals += [{"goal_spec_id": g["goal_spec_id"],
                   "ordered_subgoals": g["ordered_subgoals"]}
                  for g in spec["distinct_goals"]]
        env = make_public_env(task_name, EPISODE_LENGTH[task_name])
        try:
            obs, _ = env.reset(seed=source["seed"])
            automata = {}
            for g in goals:
                a = GoalAutomaton(g["ordered_subgoals"])
                a.start(env)
                automata[g["goal_spec_id"]] = a
            if source["provenance"] == "staged":
                stage_support(
                    env, automata[goals[0]["goal_spec_id"]], task_name,
                    recorded=source["staging_info"])
            for a in automata.values():
                a.evaluate(env, 0)
            per_decision = []
            t = 0
            for row in source["rows"]:
                entry = {"decision": row["decision"], "goals": {}}
                for gid, a in automata.items():
                    entry["goals"][gid] = {
                        "state_before": fork_env_state(a),
                        "valid_before": list(a.prev_valid),
                        "events_before": sorted(a.events_achieved),
                        "ordered_prefix_before": a.ordered_prefix(),
                    }
                for a_env in row["actions_env"]:
                    env.step(a_env)
                    t += 1
                for gid, a in automata.items():
                    a.evaluate(env, t)
                    entry["goals"][gid]["valid_after"] = list(a.prev_valid)
                err = float(np.abs(body_positions(
                    env, list(automata[goals[0][
                        "goal_spec_id"]].bodies.values()))
                    - row["obj_after"]).max())
                assert err < REREACH_ATOL, (
                    f"{source['source_id']} d={row['decision']}: label "
                    f"replay diverges {err:.2e}")
                per_decision.append(entry)
            torch.save({"schema": "v06_labels_v1",
                        "source_id": source["source_id"],
                        "goals": [g["goal_spec_id"] for g in goals],
                        "per_decision": per_decision}, out_path)
            print(f"[labels] {source['source_id']}: "
                  f"{len(per_decision)} decisions x {len(goals)} goals",
                  flush=True)
        finally:
            env.close()

    # ---- pass 2: h features per text variant ---------------------------
    from lcwm.chassis import Pi05Runner  # noqa: F811
    runner = Pi05Runner(suite_name="libero_10")
    for path in sources:
        source = torch.load(path, weights_only=False)
        task_name = source["task"]
        for variant in variants_for(task_name):
            vid = variant["variant_id"]
            out_path = (DATA / "features"
                        / f"{path.stem}__{vid}.pt")
            if out_path.exists():
                continue
            runner.reset()
            hs, masks = [], []
            for row in source["rows"]:
                batch = runner._obs_to_policy_batch(
                    row["obs"], variant["language"])
                prefix = prefix_forward(runner.policy, batch)
                hs.append(prefix.hidden[0].half().cpu())
                masks.append(prefix.pad_masks[0].bool().cpu())
            torch.save({
                "schema": "v06_features_v1",
                "source_id": source["source_id"],
                "variant": variant,
                "prompt_sha256": hashlib.sha256(
                    variant["language"].encode()).hexdigest()[:16],
                "h": torch.stack(hs),
                "mask": torch.stack(masks),
            }, out_path)
            print(f"[features] {source['source_id']} {vid}: "
                  f"{len(hs)} decisions", flush=True)
    print("feature/label cache complete", flush=True)


if __name__ == "__main__":
    main()
