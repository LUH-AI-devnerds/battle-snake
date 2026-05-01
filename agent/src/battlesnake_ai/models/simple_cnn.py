import torch
import torch.nn as nn
import numpy as np

from battlesnake_ai.models.base import BaseModel

class SimpleCNN(BaseModel):
    """
    A simple Convolutional Neural Network for Battlesnake.
    """
    def __init__(self, in_channels: int, num_actions: int = 4):
        super().__init__(in_channels, num_actions)
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_actions)
        )

    def forward(self, obs_np: np.ndarray) -> torch.Tensor:
        """
        Forward pass converting the raw numpy observation to action logits.
        """
        x = self.preprocess_observation(obs_np)
        return self.net(x)
