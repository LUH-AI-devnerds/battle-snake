"""Prioritized experience replay with proportional sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from battlesnake_ai.training.replay_buffer import Transition


@dataclass
class PrioritizedTransition(Transition):
    """Transition with n-step return and discount factor for the n-step horizon."""

    n_step_return: float = 0.0
    gamma_n: float = 1.0


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: List[Optional[PrioritizedTransition]] = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, p: float, data: PrioritizedTransition) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, PrioritizedTransition]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.data[data_idx]  # type: ignore[return-value]

    def __len__(self) -> int:
        return self.n_entries


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
    ):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0

    @property
    def capacity(self) -> int:
        return self.tree.capacity

    def __len__(self) -> int:
        return len(self.tree)

    def push(self, t: PrioritizedTransition) -> None:
        p = self.max_priority**self.alpha
        self.tree.add(p, t)

    def sample(self, batch_size: int, beta: float) -> Tuple[List[PrioritizedTransition], List[int], np.ndarray]:
        batch: List[PrioritizedTransition] = []
        indices: List[int] = []
        priorities = []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(self.tree.tree[idx])
        sampling_probs = np.asarray(priorities, dtype=np.float64) / self.tree.total()
        weights = (len(self) * sampling_probs) ** (-beta)
        weights /= weights.max()
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            p = (abs(float(err)) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, p)
            self.tree.update(idx, p)
