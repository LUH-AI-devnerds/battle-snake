import torch
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    """
    Abstract base class for Battlesnake models.
    Enforces a common interface for processing game observations.
    """
    def __init__(self, in_channels: int, num_actions: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions

    def preprocess_observation(self, obs_np: np.ndarray) -> torch.Tensor:
        """
        Converts a numpy observation of shape (Batch, Width, Height, Channels)
        into a PyTorch tensor of shape (Batch, Channels, Width, Height).
        """
        x = torch.from_numpy(obs_np).float()
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, obs_np: np.ndarray) -> torch.Tensor:
        """
        Must return logits of shape (Batch, num_actions).
        """
        raise NotImplementedError("Subclasses must implement the forward method.")
