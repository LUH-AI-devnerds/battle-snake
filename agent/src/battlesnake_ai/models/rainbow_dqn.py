"""Rainbow DQN: dueling + distributional (C51) head with shared CNN backbone."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from battlesnake_ai.models.backbone import ConvBackbone
from battlesnake_ai.models.base import BaseModel


class RainbowDQN(BaseModel):
    def __init__(
        self,
        in_channels: int,
        num_actions: int = 4,
        num_atoms: int = 51,
        v_min: float = -1.0,
        v_max: float = 1.0,
        feature_dim: int = 64,
    ):
        super().__init__(in_channels, num_actions)
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.backbone = ConvBackbone(in_channels, feature_dim=feature_dim)
        self.value_head = nn.Linear(feature_dim, num_atoms)
        self.advantage_head = nn.Linear(feature_dim, num_actions * num_atoms)
        support = torch.linspace(v_min, v_max, num_atoms)
        self.register_buffer("support", support)
        delta = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer("delta_z", torch.tensor(delta))

    def _features(self, obs_np: np.ndarray) -> torch.Tensor:
        x = self.preprocess_observation(obs_np)
        device = next(self.parameters()).device
        return self.backbone(x.to(device))

    def forward_dist(self, obs_np: np.ndarray) -> torch.Tensor:
        """Return log-probabilities over atoms: (batch, num_actions, num_atoms)."""
        features = self._features(obs_np)
        batch = features.shape[0]
        value = self.value_head(features).view(batch, 1, self.num_atoms)
        advantage = self.advantage_head(features).view(batch, self.num_actions, self.num_atoms)
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        logits = value + advantage
        return F.log_softmax(logits, dim=2)

    def forward(self, obs_np: np.ndarray) -> torch.Tensor:
        """Expected Q-values for action selection: (batch, num_actions)."""
        log_dist = self.forward_dist(obs_np)
        prob = log_dist.exp()
        q = (prob * self.support.view(1, 1, -1)).sum(dim=2)
        return q

    def q_values_numpy(self, obs_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.forward(obs_np).detach().cpu().numpy()[0]

    @torch.no_grad()
    def project_distribution(
        self,
        next_dist: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma_n: torch.Tensor,
    ) -> torch.Tensor:
        """
        Categorical algorithm projection.
        next_dist: (batch, num_actions, num_atoms) log-probs from target net at greedy actions
        rewards, dones, gamma_n: (batch,)
        Returns target log-prob (batch, num_atoms).
        """
        batch = rewards.shape[0]
        prob = next_dist.exp()
        tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma_n.unsqueeze(1) * self.support
        tz = tz.clamp(self.v_min, self.v_max)
        b = (tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        l = l.clamp(0, self.num_atoms - 1)
        u = u.clamp(0, self.num_atoms - 1)

        offset = (
            torch.arange(batch, device=prob.device)
            .unsqueeze(1)
            .expand(batch, self.num_atoms)
            * self.num_atoms
        )
        proj = torch.zeros(batch, self.num_atoms, device=prob.device)
        for i in range(self.num_atoms):
            pl = prob[:, i] * (u[:, i] - b[:, i])
            pu = prob[:, i] * (b[:, i] - l[:, i])
            proj.view(-1).index_add_(0, (l[:, i] + offset[:, i]).view(-1), pl.view(-1))
            eq = l[:, i] == u[:, i]
            pl_eq = pl.clone()
            pl_eq[eq] = 0.0
            pu_eq = pu.clone()
            pu_eq[eq] = 0.0
            proj.view(-1).index_add_(0, (l[:, i] + offset[:, i]).view(-1), pl_eq.view(-1))
            proj.view(-1).index_add_(0, (u[:, i] + offset[:, i]).view(-1), pu_eq.view(-1))

        proj = proj.clamp(min=1e-5)
        return torch.log(proj)
