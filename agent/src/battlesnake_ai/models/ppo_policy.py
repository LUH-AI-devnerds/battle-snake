"""PPO actor-critic with shared CNN backbone."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from battlesnake_ai.models.backbone import ConvBackbone
from battlesnake_ai.models.base import BaseModel


class PPOPolicy(BaseModel):
    def __init__(self, in_channels: int, num_actions: int = 4, feature_dim: int = 64):
        super().__init__(in_channels, num_actions)
        self.backbone = ConvBackbone(in_channels, feature_dim=feature_dim)
        self.actor = nn.Linear(feature_dim, num_actions)
        self.critic = nn.Linear(feature_dim, 1)

    def _features(self, obs_np: np.ndarray) -> torch.Tensor:
        x = self.preprocess_observation(obs_np)
        device = next(self.parameters()).device
        return self.backbone(x.to(device))

    def forward(self, obs_np: np.ndarray) -> torch.Tensor:
        return self.actor_logits(obs_np)

    def actor_logits(self, obs_np: np.ndarray) -> torch.Tensor:
        return self.actor(self._features(obs_np))

    def value(self, obs_np: np.ndarray) -> torch.Tensor:
        return self.critic(self._features(obs_np)).squeeze(-1)

    def act(
        self, obs_np: np.ndarray, legal_actions: list[int]
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action from masked distribution; return action, log_prob, value."""
        logits = self.actor_logits(obs_np)[0]
        mask = torch.full_like(logits, -1e9)
        for a in legal_actions:
            mask[a] = 0.0
        masked = logits + mask
        dist = torch.distributions.Categorical(logits=masked)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action), self.value(obs_np)[0]

    def evaluate_actions(
        self, obs_np: np.ndarray, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor_logits(obs_np)
        values = self.critic(self._features(obs_np)).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy
