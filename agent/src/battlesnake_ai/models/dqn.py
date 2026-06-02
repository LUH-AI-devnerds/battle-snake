import numpy as np
import torch
import torch.nn as nn

from battlesnake_ai.models.backbone import ConvBackbone
from battlesnake_ai.models.base import BaseModel


class DQN(BaseModel):
    """
    Convolutional Q-network: maps per-snake observations to Q(s, ·) for four moves.
    """

    def __init__(self, in_channels: int, num_actions: int = 4, feature_dim: int = 64):
        super().__init__(in_channels, num_actions)
        self.backbone = ConvBackbone(in_channels, feature_dim=feature_dim)
        self.head = nn.Linear(feature_dim, num_actions)

    def forward(self, obs_np: np.ndarray) -> torch.Tensor:
        x = self.preprocess_observation(obs_np)
        device = next(self.parameters()).device
        x = x.to(device)
        return self.head(self.backbone(x))
