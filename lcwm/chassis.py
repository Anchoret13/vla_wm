"""Stage-2 chassis (plan §3): raw single-env eval loop with full lerobot parity.

Parity strategy: never re-implement preprocessing. We build the SAME four pipeline
objects lerobot-eval builds (env_preprocessor → preprocessor → policy → postprocessor
→ env_postprocessor) via lerobot's own factories, and drive a single LiberoEnv
(lerobot's gym wrapper over OffScreenRenderEnv) directly. The only additions over
lerobot-eval are (a) a raw sim handle for snapshot/restore and (b) chunk-level access.

Done-check for this module: `scripts/parity_check.py` must reproduce Stage-1a numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from lcwm.libero_paths import ensure_project_libero_config

ensure_project_libero_config()  # must precede any libero.libero import (config @ import time)

from libero.libero import benchmark  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.envs.factory import make_env_pre_post_processors  # noqa: E402
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402

DEFAULT_MODEL = "lerobot/pi05_libero_finetuned"


def make_task_suite(suite_name: str):
    return benchmark.get_benchmark_dict()[suite_name]()


def make_task_env(
    suite_name: str,
    task_id: int,
    episode_index: int = 0,
    env_cfg: LiberoEnvConfig | None = None,
) -> LiberoEnv:
    """Single LiberoEnv with the SAME construction defaults lerobot-eval uses.

    We intentionally do not override observation size / cameras / control mode:
    parity means env_cfg defaults (obs 360x360; resize to policy input happens in
    the processor pipelines, not here).
    """
    cfg = env_cfg or LiberoEnvConfig(task=suite_name)
    suite = make_task_suite(suite_name)
    # Bare LiberoEnv has no gym TimeLimit wrapper; without an explicit cap the
    # loop can outrun robosuite's internal horizon, whose done is NOT surfaced
    # by bddl_base_domain.step ("executing action in terminated episode").
    # Match the vec-env eval exactly: suite-specific max steps.
    episode_length = cfg.episode_length or TASK_SUITE_MAX_STEPS[suite_name]
    return LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        episode_length=episode_length,
        camera_name=cfg.camera_name,
        obs_type=cfg.obs_type,
        observation_width=cfg.observation_width,
        observation_height=cfg.observation_height,
        init_states=cfg.init_states,
        episode_index=episode_index,
        camera_name_mapping=cfg.camera_name_mapping,
    )


class Pi05Runner:
    """Frozen pi0.5 + the exact lerobot eval pipelines, single-env."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        suite_name: str = "libero_spatial",
        device: str = "cuda",
        n_action_steps: int = 10,
    ):
        self.model_id = model_id
        self.suite_name = suite_name

        # Mirror lerobot_eval.py main(): policy config from pretrained + overrides.
        policy_cfg = PreTrainedConfig.from_pretrained(model_id)
        policy_cfg.pretrained_path = model_id
        policy_cfg.compile_model = False
        policy_cfg.n_action_steps = n_action_steps  # chunk commitment c
        policy_cfg.device = device
        self.policy_cfg = policy_cfg

        self.env_cfg = LiberoEnvConfig(task=suite_name)

        self.policy = make_policy(cfg=policy_cfg, env_cfg=self.env_cfg)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=model_id,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=self.env_cfg, policy_cfg=policy_cfg
        )

    # ---- observation plumbing -------------------------------------------------

    @staticmethod
    def _batch_leaf(v: Any) -> Any:
        """Recursively add a leading batch dim, mirroring gym's vec-env stacking
        of (arbitrarily) nested Dict observation spaces."""
        if isinstance(v, dict):
            return {k: Pi05Runner._batch_leaf(x) for k, x in v.items()}
        if isinstance(v, np.ndarray):
            return np.expand_dims(v, 0)
        if isinstance(v, (int, float, bool, np.number)):
            return np.asarray([v])
        return [v]

    @staticmethod
    def _batch_obs(obs: dict, task_description: str) -> dict:
        """Single-env obs -> the batched layout the vec-env eval loop feeds
        the pipelines."""
        batched = {k: Pi05Runner._batch_leaf(v) for k, v in obs.items()}
        batched["task"] = [task_description]
        return batched

    def _obs_to_policy_batch(self, obs: dict, task_description: str) -> dict:
        # Mirror the eval rollout exactly: batch -> preprocess_observation
        # (numpy->torch, LeRobot key layout) -> task -> env pipeline -> policy pipeline.
        observation = self._batch_obs(obs, task_description)
        observation = preprocess_observation(observation)
        observation["task"] = [task_description]
        observation = self.env_preprocessor(observation)
        observation = self.preprocessor(observation)
        return observation

    # ---- action interfaces ----------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: dict, task_description: str) -> np.ndarray:
        """One env action; the policy's internal queue re-samples every n_action_steps."""
        observation = self._obs_to_policy_batch(obs, task_description)
        action = self.policy.select_action(observation)
        action = self.postprocessor(action)
        transition = self.env_postprocessor({"action": action})
        return transition["action"][0].cpu().numpy() if torch.is_tensor(
            transition["action"]
        ) else np.asarray(transition["action"])[0]

    @torch.no_grad()
    def sample_chunk(self, obs: dict, task_description: str) -> torch.Tensor:
        """Full action chunk (1, chunk_size, 7) — Stage-4/5 entry point (pre env-postprocess)."""
        observation = self._obs_to_policy_batch(obs, task_description)
        return self.policy.predict_action_chunk(observation)

    def reset(self) -> None:
        self.policy.reset()  # clears the internal action queue


# ---- episode loop -------------------------------------------------------------


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    sum_reward: float
    seed: int | None = None
    extras: dict = field(default_factory=dict)


def run_episode(
    runner: Pi05Runner,
    env: LiberoEnv,
    seed: int | None = None,
    max_steps: int | None = None,
    step_hook: Callable[[int, dict, np.ndarray, dict], None] | None = None,
) -> EpisodeResult:
    """Parity episode: LiberoEnv handles init-state selection and the settle steps
    (num_steps_wait=10 noops) inside reset, exactly as in lerobot-eval."""
    runner.reset()
    obs, info = env.reset(seed=seed)
    task_desc = env.task_description

    done, success, total_r, t = False, False, 0.0, 0
    limit = max_steps or env.episode_length or 10_000
    while not done and t < limit:
        action = runner.select_action(obs, task_desc)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += float(reward)
        done = bool(terminated or truncated)
        success = success or bool(info.get("is_success", False)) or float(reward) > 0
        if step_hook is not None:
            step_hook(t, obs, action, info)
        t += 1
    return EpisodeResult(success=success, steps=t, sum_reward=total_r, seed=seed)
