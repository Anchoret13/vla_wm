"""V8.0 mechanism-development benchmark registry (framework v1.0 §13.1).

A NEW module, deliberately not an edit of `lcwm/loho.py`.  The historical
`CHAINS` registry there pins chain3 at 700 steps and chain5 at 990; every
V7.x record was produced under it and must stay reproducible byte-for-byte.
V8.0 uses a different, *derived* horizon, so it gets its own registry and its
own env factory.

Horizon derivation (framework §13.1, locked before any V8.0 outcome):

    episode_length = min(990, 250 * n_subgoals)

250 steps per subgoal comes from the V7.7 completion-step floor - the earliest
`pick_up` milestone in that probe occurred at step 66, and its twelve successes
spanned 66..106 - plus margin for the place phase that follows each pick.  The
990 cap is robosuite's internal horizon of 1000 (raising past it produced
"executing action in terminated episode" at step 1001; recorded in
`lcwm/loho.py`).

Because the horizon differs, **the historical Chain3 `0/5` at 700 steps is not
the baseline of this screen and must not be quoted as one.**
"""

from __future__ import annotations

from pathlib import Path

from lcwm.libero_paths import ensure_project_libero_config

ensure_project_libero_config()

from lcwm.loho import (  # noqa: E402
    BDDL_DIR,
    ChainEnv,
    _StubSuite,
    _StubTask,
    read_language,
)

HORIZON_PER_SUBGOAL = 250
ROBOSUITE_INTERNAL_HORIZON = 1000
HORIZON_CAP = 990

#: Candidate task ladder for the V8.0 screen.  Identical scene in all five
#: (same 7 objects + basket, same init regions); only the goal conjunction and
#: the instruction differ.  Two rungs are duplicated at a different object pair
#: (`chain1b`, `chain2b`) because V7.7 showed object identity is a live
#: variable - pi0.5 would not redirect to object 2 while object 1 was present -
#: so a one-object-per-rung ladder could confound difficulty with identity.
V080_TASKS: dict[str, dict] = {
    "chain1_lr2":  {"n_subgoals": 1, "objs": ["alphabet_soup_1"]},
    "chain1b_lr2": {"n_subgoals": 1, "objs": ["tomato_sauce_1"]},
    "chain2_lr2":  {"n_subgoals": 2, "objs": ["alphabet_soup_1", "tomato_sauce_1"]},
    "chain2b_lr2": {"n_subgoals": 2, "objs": ["tomato_sauce_1", "cream_cheese_1"]},
    "chain3_lr2":  {"n_subgoals": 3, "objs": ["alphabet_soup_1", "tomato_sauce_1",
                                              "cream_cheese_1"]},
}

BASKET_REGION = "basket_1_contain_region"

#: The ordered milestone list the frozen `GoalAutomaton` consumes, in the
#: repository's established interleaved convention (`scripts/build_v073_data.py`,
#: `scripts/build_goal_spec_manifest.py`): every object contributes a `pick_up`
#: EVENT milestone (sticky; displacement >= 0.02 and lift >= 0.03) and a `place`
#: PREDICATE milestone (revocable).  The BDDL goal conjunction only ever exposes
#: the `place` half, so a screen driven off goal atoms alone would be blind to
#: the milestone class that V7.7 actually measured (`pick_up`, floor ~66 steps)
#: and would give the 1-subgoal rungs a single reachable failure phase.
for _name, _spec in V080_TASKS.items():
    _spec["ordered_subgoals"] = [
        sg for o in _spec["objs"]
        for sg in (f"pick_up {o}", f"place {o} {BASKET_REGION}")
    ]
    _spec["n_milestones"] = len(_spec["ordered_subgoals"])
del _name, _spec


#: Frozen seed families.  Disjoint by construction; asserted at run time.
SEED_FAMILIES = {
    "screen":    list(range(3000, 3010)),   # 10 seeds/task
    "confirm":   list(range(3100, 3130)),   # 30 seeds/task, disjoint from screen
    "behavior":  list(range(3200, 3260)),   # RESERVED - never executed in V8.0
    "acquisition": list(range(3300, 3400)),  # RESERVED - never executed in V8.0
}

EXECUTABLE_PANELS = ("screen", "confirm")


def episode_length(name: str) -> int:
    """Derived horizon for a V8.0 task, capped at the robosuite limit."""
    n = V080_TASKS[name]["n_subgoals"]
    return min(HORIZON_CAP, HORIZON_PER_SUBGOAL * n)


def bddl_path(name: str) -> Path:
    return BDDL_DIR / f"{name}.bddl"


def make_v080_env(name: str) -> ChainEnv:
    """`lcwm.loho.make_chain_env` with the V8.0 derived horizon.

    Mirrors that constructor exactly (obs_type / resolution / cameras from the
    LiberoEnvConfig defaults); the only difference is where `episode_length`
    comes from.  ChainEnv is used rather than a bare LiberoEnv because the
    parent auto-resets on termination and would corrupt terminal predicates.
    """
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    if name not in V080_TASKS:
        raise KeyError(f"{name!r} is not a V8.0 task: {sorted(V080_TASKS)}")
    path = bddl_path(name)
    if not path.exists():
        raise FileNotFoundError(path)

    cfg = LiberoEnvConfig(task="libero_10")
    task = _StubTask(name, read_language(name))
    env = ChainEnv(
        task_suite=_StubSuite(task), task_id=0, task_suite_name="libero_10",
        episode_length=episode_length(name),
        camera_name=cfg.camera_name,
        obs_type=cfg.obs_type,
        observation_width=cfg.observation_width,
        observation_height=cfg.observation_height,
        init_states=False,
        camera_name_mapping=cfg.camera_name_mapping,
    )
    env._task_bddl_file = str(path)
    return env


def assert_seed_families_disjoint() -> None:
    seen: dict[int, str] = {}
    for fam, seeds in SEED_FAMILIES.items():
        for s in seeds:
            if s in seen:
                raise AssertionError(
                    f"seed {s} appears in both {seen[s]!r} and {fam!r}")
            seen[s] = fam


assert_seed_families_disjoint()


# --------------------------------------------------------------------------
# V8.0 episode runner
# --------------------------------------------------------------------------
# `lcwm.loho.run_chain_episode` is left untouched: every V7.x record was
# produced by it and its file hash is cited in sealed manifests.  This is the
# same loop with one addition - the frozen `GoalAutomaton` is evaluated at the
# same stride, so the screen measures progress with the SAME p(t) that
# framework §3.1.1 will use to place stall anchors, instead of the goal-atom
# conjunction which exposes only the `place` half of each subgoal.

from dataclasses import dataclass, field  # noqa: E402

from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402


@dataclass
class V080Result:
    task: str
    seed: int
    condition: str
    # LIBERO semantics: success is exactly the conjunction of goal atoms.
    success: bool
    q_score: float
    steps: int
    instruction: str
    # automaton-side progress, the quantity §3.1.1 / §5.2.1 are defined over
    subgoals: list[str] = field(default_factory=list)
    events_achieved: dict[int, int] = field(default_factory=dict)
    flips: list = field(default_factory=list)
    milestone_timeline: list = field(default_factory=list)
    ordered_prefix: int = 0
    q_valid: float = 0.0
    damage_unrecovered: int = 0
    # goal-atom timeline, retained for comparability with the V7.x records
    bits_timeline: list = field(default_factory=list)
    subgoal_completion_t: dict = field(default_factory=dict)


def run_v080_episode(runner, env, seed: int, stride: int = 10) -> V080Result:
    """One full-prompt `N=1` stock rollout with automaton progress recording.

    The episode seed is applied to torch/numpy as well as to the env: pi0.5 is
    a flow policy and samples its action chunk, so without this two runs of the
    same seed differ (observed: chain1 seed 3000 terminated at 138 and 134
    steps on two runs of the same code).  A calibration whose panel cannot be
    reproduced is not a calibration.
    """
    import numpy as _np
    import torch as _torch

    _torch.manual_seed(seed)
    _np.random.seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

    runner.reset()
    obs, _ = env.reset(seed=seed)

    spec = V080_TASKS[env.task] if env.task in V080_TASKS else None
    subgoals = spec["ordered_subgoals"] if spec else []
    automaton = GoalAutomaton(subgoals)
    automaton.start(env)          # must follow reset: start_pos is the baseline

    atoms = goal_atoms(env)
    instruction = env.task_description
    bits = predicate_bits(env, atoms)
    completion: dict[str, int] = {}
    bits_timeline = [(0, bits.tolist())]
    automaton.evaluate(env, 0)

    done, success_from_info, t = False, False, 0
    limit = env.episode_length
    while not done and t < limit:
        action = runner.select_action(obs, instruction)
        obs, _r, term, trunc, step_info = env.step(action)
        t += 1
        done = bool(term or trunc)
        success_from_info = (success_from_info
                             or bool(step_info.get("is_success", False)))
        if t % stride == 0 or done:
            # ChainEnv never auto-resets, so the simulator still holds the
            # terminal scene here and both readouts are valid.
            automaton.evaluate(env, t)
            new_bits = predicate_bits(env, atoms)
            for a, old, new in zip(atoms, bits, new_bits):
                if new and not old and a[1] not in completion:
                    completion[a[1]] = t
            bits = new_bits
            bits_timeline.append((t, bits.tolist()))

    return V080Result(
        task=env.task, seed=seed, condition="full",
        success=bool(success_from_info or bits.all()),
        q_score=float(bits.mean()), steps=t, instruction=instruction,
        subgoals=list(subgoals),
        events_achieved={int(k): int(v) for k, v in automaton.events_achieved.items()},
        flips=[list(f) for f in automaton.flips],
        milestone_timeline=[(s, list(v)) for s, v in automaton.timeline],
        ordered_prefix=automaton.ordered_prefix(),
        q_valid=automaton.q_valid(),
        damage_unrecovered=automaton.damage_unrecovered(),
        bits_timeline=bits_timeline,
        subgoal_completion_t={k: int(v) for k, v in completion.items()},
    )
