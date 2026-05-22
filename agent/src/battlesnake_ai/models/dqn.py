import numpy as np
import torch
import torch.nn as nn

from battlesnake_ai.models.base import BaseModel


class DQN(BaseModel):
    """
    Convolutional Q-network: maps per-snake observations to Q(s, ·) for four moves.
    """

    def __init__(self, in_channels: int, num_actions: int = 4):
        super().__init__(in_channels, num_actions)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_actions),
        )

    def forward(self, obs_np: np.ndarray) -> torch.Tensor:
        x = self.preprocess_observation(obs_np)
        device = next(self.parameters()).device
        x = x.to(device)
        return self.net(x)
