"""Deep residual convolutional encoder for Battlesnake RL models.

Architecture:
  Conv(in→32) → 2×ResBlock(32) → Conv(32→64,stride=2) → 2×ResBlock(64) →
  Conv(64→128,stride=2) → 2×ResBlock(128) → SE attention →
  AdaptiveAvgPool(3×3) → flatten → Linear → ReLU → feature_dim

This replaces the original shallow 2-layer CNN that used global average pooling
(destroying all spatial information).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN→ReLU→Conv→BN→ReLU→Conv + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + residual


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))  # (b, c) global avg pool
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class ConvBackbone(nn.Module):
    """Deep residual encoder: 6 res blocks + SE attention → feature_dim vector.

    Compatible API with the old shallow backbone (same __init__ signature and
    forward signature).
    """

    def __init__(self, in_channels: int, feature_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Stage 1: in→32, 2 residual blocks (spatial: 29×29)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
        )

        # Stage 2: 32→64 with stride-2 downsample, 2 residual blocks (spatial: 15×15)
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # Stage 3: 64→128 with stride-2 downsample, 2 residual blocks (spatial: 8×8)
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # Channel attention after last stage
        self.se = SqueezeExcitation(128, reduction=4)

        # Spatial reduction: preserve some spatial structure → flatten → project
        self.pool = nn.AdaptiveAvgPool2d((3, 3))  # 128 * 3 * 3 = 1152
        self._flat_dim = 128 * 3 * 3
        self.fc = nn.Linear(self._flat_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) → (N, feature_dim)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        x = self.stage3(x)
        x = self.se(x)
        x = self.pool(x)
        x = x.flatten(1)
        return F.relu(self.fc(x))
