"""Chained long-horizon exam (self-built LIBERO-LoHo equivalent).

Custom chained-goal BDDL tasks on scenes INSIDE pi05_libero_finetuned's
training distribution (construction condition from the 2026-07-22 discussion:
chains on known scenes/objects — frozen policy, zero finetuning).

Two conditions per task:
  full    the whole chained instruction as-is (paper's zero-shot floor analog)
  decomp  oracle subgoal decomposition: feed one in-distribution subgoal
          instruction at a time ("pick up the X and place it in the basket",
          libero_object phrasing); switch — and reset the policy's action
          queue — when the current subgoal's predicate flips true.

Q-Score = fraction of goal atoms true at episode end (matches the paper's
metric and our per-atom machinery). SR = all atoms true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lcwm.libero_paths import ensure_project_libero_config

ensure_project_libero_config()

from lcwm.chassis import Pi05Runner  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lerobot.envs.libero import LiberoEnv  # noqa: E402

BDDL_DIR = Path(__file__).resolve().parent.parent / "bddl" / "chains"

CHAINS = {
    "chain3_lr2": {"episode_length": 700, "n_subgoals": 3},
    "chain4_lr2": {"episode_length": 900, "n_subgoals": 4},
    # 990 not the planned 1100: robosuite internal horizon is 1000
    # (1100 raised "terminated episode" at step 1001). Recorded as-run.
    "chain5_lr2": {"episode_length": 990, "n_subgoals": 5},
}

SUBGOAL_PHRASE = {
    "alphabet_soup_1": "pick up the alphabet soup and place it in the basket",
    "tomato_sauce_1": "pick up the tomato sauce and place it in the basket",
    "cream_cheese_1": "pick up the cream cheese and place it in the basket",
    "butter_1": "pick up the butter and place it in the basket",
    "milk_1": "pick up the milk and place it in the basket",
}


class ChainEnv(LiberoEnv):
    """LiberoEnv WITHOUT the auto-reset-on-termination behavior.

    Known-problem ledger 2026-07-23 (confirmed): the parent's step() calls
    self.reset() the moment `done or is_success` fires, so any terminal-state
    readout (predicate bits, Q-score) scored the freshly RESET scene. Two
    chain-exam episodes recorded q=0.00 that were in fact terminal successes.
    Here the evaluator owns reset; step() never resets.
    """

    def step(self, action):
        self._ensure_env()
        if action.ndim != 1:
            raise ValueError(f"expected 1-D action, got {action.shape}")
        raw_obs, reward, done, info = self._env.step(action)
        is_success = self._env.check_success()
        terminated = bool(done or is_success)
        info.update({"task": self.task, "task_id": self.task_id,
                     "done": done, "is_success": is_success})
        observation = self._format_raw_obs(raw_obs)
        return observation, reward, terminated, False, info


class _StubTask:
    def __init__(self, name: str, language: str):
        self.name, self.language = name, language
        self.problem_folder, self.bddl_file = "chains", f"{name}.bddl"
        self.init_states_file = ""


class _StubSuite:
    def __init__(self, task):
        self._task = task
        self.n_tasks = 1

    def get_task(self, i):
        return self._task


def read_language(name: str) -> str:
    import re
    txt = (BDDL_DIR / f"{name}.bddl").read_text()
    return re.search(r"\(:language ([^)]*)\)", txt).group(1).strip()


def make_chain_env(name: str) -> LiberoEnv:
    """Mirror chassis.make_task_env's construction EXACTLY (obs_type,
    resolution, cameras from LiberoEnvConfig defaults) — the smoke failure
    'State is required for PI05' came from ctor defaults (pixels, 256px)
    diverging from the eval contract (pixels_agent_pos, 360px)."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    cfg = LiberoEnvConfig(task="libero_10")
    lang = read_language(name)
    task = _StubTask(name, lang)
    env = ChainEnv(
        task_suite=_StubSuite(task), task_id=0, task_suite_name="libero_10",
        episode_length=CHAINS[name]["episode_length"],
        camera_name=cfg.camera_name,
        obs_type=cfg.obs_type,
        observation_width=cfg.observation_width,
        observation_height=cfg.observation_height,
        init_states=False,
        camera_name_mapping=cfg.camera_name_mapping,
    )
    env._task_bddl_file = str(BDDL_DIR / f"{name}.bddl")  # bypass suite path
    return env


@dataclass
class ChainResult:
    task: str
    condition: str
    seed: int
    success: bool
    q_score: float
    steps: int
    bits_timeline: list = field(default_factory=list)   # (t, bits) at stride
    subgoal_completion_t: dict = field(default_factory=dict)
    instructions_used: list = field(default_factory=list)


def run_chain_episode(runner: Pi05Runner, env: LiberoEnv, condition: str,
                      seed: int, stride: int = 10) -> ChainResult:
    runner.reset()
    obs, _ = env.reset(seed=seed)
    atoms = goal_atoms(env)
    full_lang = env.task_description
    bits = predicate_bits(env, atoms)
    completion = {}
    timeline = [(0, bits.tolist())]
    instructions = []

    def current_instruction():
        if condition == "full":
            return full_lang
        for a, b in zip(atoms, bits):
            if not b:
                return SUBGOAL_PHRASE[a[1]]
        return full_lang

    instr = current_instruction()
    instructions.append(instr)
    done, t = False, 0
    limit = env.episode_length
    while not done and t < limit:
        action = runner.select_action(obs, instr)
        obs, _r, term, trunc, _i = env.step(action)
        t += 1
        done = bool(term or trunc)
        if t % stride == 0 or done:
            new_bits = predicate_bits(env, atoms)
            for i, (a, old, new) in enumerate(zip(atoms, bits, new_bits)):
                if new and not old and a[1] not in completion:
                    completion[a[1]] = t
            if condition == "decomp" and (new_bits != bits).any():
                bits = new_bits
                nxt = current_instruction()
                if nxt != instr:
                    instr = nxt
                    instructions.append(instr)
                    runner.reset()          # clear committed action queue
            bits = new_bits
            timeline.append((t, bits.tolist()))

    final_bits = predicate_bits(env, atoms)
    return ChainResult(
        task=env.task, condition=condition, seed=seed,
        success=bool(final_bits.all()),
        q_score=float(final_bits.mean()),
        steps=t, bits_timeline=timeline,
        subgoal_completion_t=completion, instructions_used=instructions,
    )
