"""Pool of past opponents for self-play training."""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class OpponentPool:
    """Maintains a pool of past policy snapshots and tracks their ELO ratings."""

    def __init__(self, max_size: int = 10, initial_elo: float = 1200.0, device: Optional[torch.device] = None):
        self.max_size = max_size
        self.initial_elo = initial_elo
        self.device = device
        self.snapshots: List[Dict[str, torch.Tensor]] = []
        self.elos: List[float] = []
        self.episodes_played: List[int] = []
        self._cached_opponent: Optional[nn.Module] = None

    def add_snapshot(self, model: nn.Module) -> None:
        """Add a snapshot of the current model to the pool."""
        state_dict = copy.deepcopy(model.state_dict())
        # Move state dict to CPU to save GPU memory
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()

        if len(self.snapshots) >= self.max_size:
            # Remove lowest ELO snapshot to make room
            min_idx = self.elos.index(min(self.elos))
            self.snapshots.pop(min_idx)
            self.elos.pop(min_idx)
            self.episodes_played.pop(min_idx)

        self.snapshots.append(state_dict)
        self.elos.append(self.initial_elo)
        self.episodes_played.append(0)

    def sample_opponent(self, base_model: nn.Module) -> Optional[Tuple[nn.Module, int]]:
        """Sample an opponent from the pool based on ELO or uniformly.
        Returns the instantiated model and its index in the pool.
        Reuses a cached model shell to avoid repeated deepcopy allocations."""
        if not self.snapshots:
            return None

        # Simple uniform sampling for now
        idx = random.randint(0, len(self.snapshots) - 1)

        # Reuse a single cached shell instead of deepcopy every call
        if self._cached_opponent is None:
            self._cached_opponent = copy.deepcopy(base_model)
        self._cached_opponent.load_state_dict(self.snapshots[idx])
        if self.device:
            self._cached_opponent.to(self.device)
        self._cached_opponent.eval()

        self.episodes_played[idx] += 1
        return self._cached_opponent, idx

    def update_elo(self, idx: int, outcome: float, policy_elo: float, k_factor: float = 32.0) -> float:
        """Update ELO rating for an opponent after a game.
        outcome: 1.0 for win, 0.5 for draw, 0.0 for loss (from the pool opponent's perspective).
        Returns the new ELO of the pool opponent."""
        expected = 1.0 / (1.0 + 10 ** ((policy_elo - self.elos[idx]) / 400.0))
        self.elos[idx] = self.elos[idx] + k_factor * (outcome - expected)
        return self.elos[idx]
