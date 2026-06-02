"""Ensemble inference combining Rainbow Q-values and PPO policy logits."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch

from battlesnake_ai.inference.agent_loader import load_agent
from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.models.rainbow_dqn import RainbowDQN
from battlesnake_ai.training.action_selection import ensemble_joint


class EnsembleAgent:
    def __init__(
        self,
        rainbow: RainbowDQN,
        ppo: PPOPolicy,
        *,
        w_rainbow: float = 0.5,
        w_ppo: float = 0.5,
        device: torch.device | None = None,
    ):
        self.rainbow = rainbow
        self.ppo = ppo
        self.w_rainbow = w_rainbow
        self.w_ppo = w_ppo
        self.device = device or torch.device("cpu")
        self.rainbow.to(self.device).eval()
        self.ppo.to(self.device).eval()

    @classmethod
    def from_checkpoints(
        cls,
        rainbow_path: str,
        ppo_path: str,
        *,
        w_rainbow: float = 0.5,
        w_ppo: float = 0.5,
        device: torch.device | None = None,
    ) -> "EnsembleAgent":
        dev = device or torch.device("cpu")
        rainbow, _ = load_agent(rainbow_path, device=dev)
        ppo, _ = load_agent(ppo_path, device=dev)
        if not isinstance(rainbow, RainbowDQN):
            raise TypeError(f"Expected RainbowDQN checkpoint, got {type(rainbow)}")
        if not isinstance(ppo, PPOPolicy):
            raise TypeError(f"Expected PPOPolicy checkpoint, got {type(ppo)}")
        return cls(rainbow, ppo, w_rainbow=w_rainbow, w_ppo=w_ppo, device=dev)

    def select_joint_actions(self, env: Any, obs: np.ndarray) -> Tuple[int, ...]:
        def rainbow_scores(sl: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return self.rainbow.forward(sl).detach().cpu().numpy()[0]

        def ppo_scores(sl: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return self.ppo.actor_logits(sl).detach().cpu().numpy()[0]

        return ensemble_joint(
            env,
            obs,
            [
                (self.w_rainbow, rainbow_scores),
                (self.w_ppo, ppo_scores),
            ],
        )
