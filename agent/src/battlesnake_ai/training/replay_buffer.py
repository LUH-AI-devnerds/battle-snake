from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size FIFO replay buffer for DQN."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._buf: Deque[Transition] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        idx = np.random.choice(len(self._buf), size=batch_size, replace=False)
        items = list(self._buf)
        return [items[i] for i in idx]
