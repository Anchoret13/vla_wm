"""Specification-based LIBERO-LoHo reimplementation — public-task protocol
(H7.1; provenance and Q semantics in
results/libero_loho_public_v1/task_source_manifest.json).

Public Q-score = completed ORDERED subgoals / registered task length.
Subgoal kinds and their evaluation:
- "place OBJ REGION"  -> BDDL predicate (In OBJ REGION) via the problem
  env's _eval_predicate (same machinery as the local curriculum);
- "open FIXTURE_REGION" / "close FIXTURE_REGION" -> BDDL Open/Close
  predicate;
- "pick_up OBJ" -> registered geometric criterion: the object is displaced
  >= 0.02 m from its episode-start pose AND its z exceeds start by
  >= 0.03 m at some decision boundary (the A1 attachment/lift constants;
  a pick that was later undone still counts as completed once).
Subgoals complete in ANY temporal order physically, but Q credits a
subgoal only when all its predecessors in the registered ordered list are
also complete at or before the same boundary? NO — the public metric is
"completed subgoals"; we record BOTH: unordered completed count (primary
Q, matching the paper's fraction-of-subgoals reading) and the ordered
prefix as a declared diagnostic. This choice is registered in the
manifest and revisitable on protocol-parity evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lcwm.libero_paths import ensure_project_libero_config

ensure_project_libero_config()

from lcwm.loho import ChainEnv, _StubSuite, _StubTask  # noqa: E402
from lcwm.probe_data import body_positions, discover_object_bodies  # noqa: E402
from lcwm.seq_data import _problem_env  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_BDDL_DIR = REPO_ROOT / "bddl" / "libero_loho_public_v1"
MANIFEST_PATH = (
    REPO_ROOT / "results" / "libero_loho_public_v1"
    / "task_source_manifest.json"
)
PICK_DISPLACEMENT = 0.02
PICK_LIFT = 0.03


def load_public_tasks(status: str = "reconstructed") -> dict[str, dict]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return {
        name: spec
        for name, spec in manifest["tasks"].items()
        if spec.get("status") == status
    }


def make_public_env(task_name: str, episode_length: int = 900) -> ChainEnv:
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    import re

    bddl = PUBLIC_BDDL_DIR / f"{task_name}.bddl"
    text = bddl.read_text()
    lang = re.search(r"\(:language ([^)]*)\)", text).group(1).strip()
    cfg = LiberoEnvConfig(task="libero_10")
    task = _StubTask(task_name, lang)
    env = ChainEnv(
        task_suite=_StubSuite(task), task_id=0, task_suite_name="libero_10",
        episode_length=episode_length,
        camera_name=cfg.camera_name,
        obs_type=cfg.obs_type,
        observation_width=cfg.observation_width,
        observation_height=cfg.observation_height,
        init_states=False,
        camera_name_mapping=cfg.camera_name_mapping,
    )
    env._task_bddl_file = str(bddl)
    return env


@dataclass
class SubgoalTracker:
    """Tracks the registered ordered subgoal list of one public task."""

    subgoals: list[str]
    object_names: dict[str, str] = field(default_factory=dict)
    start_pos: dict[str, np.ndarray] = field(default_factory=dict)
    completed: dict[int, int] = field(default_factory=dict)  # idx -> step

    def start(self, env) -> None:
        self.bodies = discover_object_bodies(env)
        names = [
            s.split()[1] for s in self.subgoals if s.startswith("pick_up")
        ]
        for name in names:
            self.start_pos[name] = body_positions(
                env, [self.bodies[name]]
            )[0].copy()

    def update(self, env, step: int) -> None:
        inner = _problem_env(env)
        for index, subgoal in enumerate(self.subgoals):
            if index in self.completed:
                continue
            parts = subgoal.split()
            kind = parts[0]
            done = False
            if kind == "place":
                # Predicate function names are lowercase in LIBERO's
                # VALIDATE_PREDICATE_FN_DICT; 'place ... top_side' regions
                # are On-semantics, contain regions are In-semantics.
                predicate = "on" if parts[2].endswith("top_side") else "in"
                done = bool(
                    inner._eval_predicate([predicate, parts[1], parts[2]])
                )
            elif kind in ("open", "close"):
                done = bool(inner._eval_predicate([kind, parts[1]]))
            elif kind == "pick_up":
                pos = body_positions(env, [self.bodies[parts[1]]])[0]
                start = self.start_pos[parts[1]]
                done = (
                    float(np.linalg.norm(pos - start)) >= PICK_DISPLACEMENT
                    and float(pos[2] - start[2]) >= PICK_LIFT
                )
            if done:
                self.completed[index] = step

    def q_score(self) -> float:
        return len(self.completed) / len(self.subgoals)

    def ordered_prefix(self) -> int:
        depth = 0
        for index in range(len(self.subgoals)):
            if index in self.completed:
                depth += 1
            else:
                break
        return depth


def run_public_episode(
    runner, env, subgoals: list[str], seed: int, stride: int = 10
):
    """Full-prompt N=1 episode on a public task; returns metrics dict."""
    runner.reset()
    obs, _ = env.reset(seed=seed)
    tracker = SubgoalTracker(subgoals)
    tracker.start(env)
    instruction = env.task_description
    inner = _problem_env(env)
    goal_atoms = [list(a) for a in inner.parsed_problem["goal_state"]]
    done, t = False, 0
    while not done and t < env.episode_length:
        action = runner.select_action(obs, instruction)
        obs, _r, terminated, truncated, _info = env.step(action)
        t += 1
        done = bool(terminated or truncated)
        if t % stride == 0 or done:
            tracker.update(env, t)
    terminal_success = all(
        bool(inner._eval_predicate(a)) for a in goal_atoms
    )
    return {
        "success": bool(terminal_success),
        "q_public": tracker.q_score(),
        "subgoal_completion_steps": {
            subgoals[i]: s for i, s in sorted(tracker.completed.items())
        },
        "ordered_prefix": tracker.ordered_prefix(),
        "first_unresolved_subgoal": next(
            (
                subgoals[i]
                for i in range(len(subgoals))
                if i not in tracker.completed
            ),
            None,
        ),
        "steps": t,
        "seed": seed,
    }
