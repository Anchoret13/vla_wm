"""Stage 1R.1/1R.2 execution substrate (framework §14, daily 2026-08-19 §6).

A NEW module.  `lcwm/v080_bench.py` is sealed by the 1R.0 registration
(`V080R_REGISTRATION.json`), so it is not edited here: adding a stage must not
show up as source drift on the artifact that stage cites.

Two things this adds over the V8.0 runner, both required by the audit:

* **explicit sets.**  V8.0 stored `current_valid` and `damaged` only derivably,
  inside `milestone_timeline` and `flips`.  §14.2 requires them stored, because
  every downstream aggregation keys on them.
* **deadline separation.**  An episode runs to an EXTENDED horizon while the
  frozen deployment deadline `L` still governs what counts.  Outcomes at
  `step <= L` are official; later ones are retained and labelled
  `quarantined_post_deadline`.  A success first observed after `L` is a *late
  conversion*, which is the quantity Stage 1R.1 measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lcwm.libero_paths import ensure_project_libero_config

ensure_project_libero_config()

from lcwm.loho import BDDL_DIR, ChainEnv, _StubSuite, _StubTask, read_language  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS  # noqa: E402
from lcwm.v08r_contract import (CHECKPOINTS, Officiality, deadline,  # noqa: E402
                                officiality, stratum_key)
from lcwm.v08r_contract import extended_horizon as _extended  # noqa: E402

ROBOSUITE_INTERNAL_HORIZON = 1000


def extended_horizon(task: str) -> int:
    h = _extended(task)
    if h >= ROBOSUITE_INTERNAL_HORIZON:
        raise ValueError(f"{task}: extended horizon {h} hits the robosuite "
                         f"internal horizon {ROBOSUITE_INTERNAL_HORIZON}")
    return h


def make_env_at(task: str, episode_length: int) -> ChainEnv:
    """`make_v080_env` with an explicit horizon, constructed identically."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    if task not in CHECKPOINTS:
        raise KeyError(f"{task!r} is not a 1R task: {sorted(CHECKPOINTS)}")
    path = Path(BDDL_DIR) / f"{task}.bddl"
    if not path.exists():
        raise FileNotFoundError(path)
    cfg = LiberoEnvConfig(task="libero_10")
    stub = _StubTask(task, read_language(task))
    env = ChainEnv(
        task_suite=_StubSuite(stub), task_id=0, task_suite_name="libero_10",
        episode_length=episode_length, camera_name=cfg.camera_name,
        obs_type=cfg.obs_type, observation_width=cfg.observation_width,
        observation_height=cfg.observation_height, init_states=False,
        camera_name_mapping=cfg.camera_name_mapping)
    env._task_bddl_file = str(path)
    return env


@dataclass
class DeadlineEpisode:
    task: str
    seed: int
    subrole: str
    deadline: int
    horizon_run: int
    steps: int
    terminated_reason: str
    success_step: int | None
    success_official: bool
    late_conversion: bool
    officiality: str
    subgoals: list[str] = field(default_factory=list)
    events_achieved: dict[int, int] = field(default_factory=dict)
    current_valid: list[int] = field(default_factory=list)   # explicit (§14.2)
    damaged: list[int] = field(default_factory=list)         # explicit (§14.2)
    stratum_terminal: dict = field(default_factory=dict)
    at_checkpoint: dict = field(default_factory=dict)
    milestone_timeline: list = field(default_factory=list)
    flips: list = field(default_factory=list)
    instruction: str = ""
    wall_s: float = 0.0


def _sets_at(timeline, events, subgoals, flips, upto: int):
    """Reconstruct (ever_achieved, current_valid, damaged) as of step `upto`."""
    ever = sorted(i for i, s in events.items() if s <= upto)
    valid: list[int] = []
    for step, vals in timeline:
        if step <= upto:
            valid = [i for i, v in enumerate(vals) if v]
    dropped = {i for (s, i, d) in flips if d == -1 and s <= upto}
    return ever, valid, sorted(dropped - set(valid))


def run_deadline_episode(runner, env, task: str, seed: int, subrole: str,
                         stride: int = 10) -> DeadlineEpisode:
    """One stock full-prompt `N=1` rollout to `env.episode_length`.

    The frozen deadline governs officiality; the horizon only governs how long
    we are allowed to keep watching.
    """
    import time

    import numpy as _np
    import torch as _torch

    _torch.manual_seed(seed)
    _np.random.seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    runner.reset()
    obs, _ = env.reset(seed=seed)

    subgoals = V080_TASKS[task]["ordered_subgoals"]
    automaton = GoalAutomaton(subgoals)
    automaton.start(env)
    atoms = goal_atoms(env)
    instruction = env.task_description
    bits = predicate_bits(env, atoms)
    automaton.evaluate(env, 0)

    L = deadline(task)
    limit = env.episode_length
    done, success_step, t = False, None, 0
    reason = "horizon"
    while not done and t < limit:
        obs, _r, term, trunc, info = env.step(runner.select_action(obs, instruction))
        t += 1
        done = bool(term or trunc)
        if success_step is None and bool(info.get("is_success", False)):
            success_step = t
        if t % stride == 0 or done:
            automaton.evaluate(env, t)
            bits = predicate_bits(env, atoms)
            if success_step is None and bits.all():
                success_step = t
        if done:
            reason = "success" if success_step is not None else "terminated"

    timeline = [(s, list(v)) for s, v in automaton.timeline]
    flips = [list(f) for f in automaton.flips]
    events = {int(k): int(v) for k, v in automaton.events_achieved.items()}

    at_ckpt = {}
    for h in CHECKPOINTS[task]:
        ever, valid, dmg = _sets_at(timeline, events, subgoals, flips, h)
        at_ckpt[str(h)] = {
            "success": bool(success_step is not None and success_step <= h),
            "ever_achieved": ever, "current_valid": valid, "damaged": dmg,
            "stratum": stratum_key(task, subgoals, ever, valid, dmg).to_dict(),
            "officiality": officiality(h, L).value,
        }

    ever_t, valid_t, dmg_t = _sets_at(timeline, events, subgoals, flips, t)
    off = (officiality(success_step, L).value if success_step is not None
           else officiality(t, L).value)
    return DeadlineEpisode(
        task=task, seed=seed, subrole=subrole, deadline=L, horizon_run=limit,
        steps=t, terminated_reason=reason, success_step=success_step,
        success_official=bool(success_step is not None and success_step <= L),
        late_conversion=bool(success_step is not None and success_step > L),
        officiality=off, subgoals=list(subgoals), events_achieved=events,
        current_valid=valid_t, damaged=dmg_t,
        stratum_terminal=stratum_key(task, subgoals, ever_t, valid_t, dmg_t).to_dict(),
        at_checkpoint=at_ckpt, milestone_timeline=timeline, flips=flips,
        instruction=instruction, wall_s=round(time.time() - t0, 2))


def prefix_identity(extended: DeadlineEpisode, duplicate: DeadlineEpisode
                    ) -> tuple[bool, dict]:
    """Exact-equality duplicate check (1R.0 RNG seal, tolerance 0).

    The duplicate runs to the ORIGINAL horizon `L`; the extended run may go
    further.  Comparable quantities are therefore the extended run truncated at
    `L`.  Any mismatch is a provenance HALT, not a tolerance to widen.
    """
    L = duplicate.horizon_run
    ext_events = {i: s for i, s in extended.events_achieved.items() if s <= L}
    dup_events = {i: s for i, s in duplicate.events_achieved.items() if s <= L}
    ext_steps = min(extended.steps, L)
    ext_succ = bool(extended.success_step is not None and extended.success_step <= L)
    detail = {
        "events_extended_truncated": {str(k): v for k, v in sorted(ext_events.items())},
        "events_duplicate": {str(k): v for k, v in sorted(dup_events.items())},
        "terminal_step_extended_truncated": ext_steps,
        "terminal_step_duplicate": duplicate.steps,
        "success_extended_truncated": ext_succ,
        "success_duplicate": duplicate.success_official,
    }
    ok = (ext_events == dup_events
          and ext_steps == duplicate.steps
          and ext_succ == duplicate.success_official)
    detail["match"] = ok
    return ok, detail
