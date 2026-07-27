"""A2 — replay policy interface over cached raw observations (v0.4 P5 needs).

Given a `chain_episode_cache_v1` episode and a decision index, rebuild the
exact policy-side state without a live environment:

- `prefix_at(...)`: obs dict → `_obs_to_policy_batch` → `prefix_forward`
  → PrefixCache (hidden + KV) under the episode's instruction or ANY
  alternative instruction (crossed-language recomputation);
- `candidates_at(...)`: N π0.5-supported 50-step chunks with explicit seeds
  (candidate generation for model-guided distillation);
- `verify_prefix(...)`: regenerated hidden must match the cached hidden —
  the A2 exit criterion's exactness check.
"""

from __future__ import annotations

import torch

from lcwm.sampler import prefix_forward, sample_chunks


class ReplayPolicyInterface:
    def __init__(self, runner):
        self.runner = runner

    @torch.no_grad()
    def batch_at(
        self, episode: dict, decision: int, instruction: str | None = None
    ) -> dict:
        obs = episode["raw_observations"][decision]
        language = instruction or episode["language"]
        return self.runner._obs_to_policy_batch(obs, language)

    @torch.no_grad()
    def prefix_at(
        self, episode: dict, decision: int, instruction: str | None = None
    ):
        return prefix_forward(
            self.runner.policy, self.batch_at(episode, decision, instruction)
        )

    @torch.no_grad()
    def candidates_at(
        self,
        episode: dict,
        decision: int,
        n: int,
        seed: int,
        instruction: str | None = None,
    ) -> torch.Tensor:
        """[n, 50, 7] normalized π0.5-supported candidate chunks."""
        batch = self.batch_at(episode, decision, instruction)
        prefix = prefix_forward(self.runner.policy, batch)
        return sample_chunks(
            self.runner.policy, batch, n=n, seed=seed, prefix=prefix
        )

    @torch.no_grad()
    def verify_prefix(
        self, episode: dict, decision: int, atol: float = 2e-3
    ) -> float:
        """Max |regenerated − cached| prefix hidden; raises above atol.

        The cache stores fp16; regeneration runs the same frozen trunk on
        the same stored observation, so agreement should be at fp16
        rounding level."""
        regenerated = self.prefix_at(episode, decision)
        cached = episode["prefix_hidden"][decision].to(
            regenerated.hidden.device, torch.float32
        )
        error = float(
            (regenerated.hidden[0].float() - cached).abs().max()
        )
        if error > atol:
            raise AssertionError(
                f"decision {decision}: regenerated prefix deviates "
                f"{error:.3e} > {atol:.1e} from cache"
            )
        return error
