"""Shared convolutional encoder for Battlesnake RL models."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBackbone(nn.Module):
    """Conv2d stack → 64-d feature vector per observation."""

    def __init__(self, in_channels: int, feature_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self._out = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) → (N, feature_dim)."""
        return self._out(self.net(x))
