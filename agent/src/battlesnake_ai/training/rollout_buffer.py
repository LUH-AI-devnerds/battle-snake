"""On-policy rollout storage for PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RolloutStep:
    obs: np.ndarray
    action: int
    log_prob: float
    reward: float
    done: bool
    value: float


class RolloutBuffer:
    def __init__(self) -> None:
        self.steps: List[RolloutStep] = []

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)

    def push(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def compute_gae(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.array([s.reward for s in self.steps], dtype=np.float32)
        values = np.array([s.value for s in self.steps], dtype=np.float32)
        dones = np.array([s.done for s in self.steps], dtype=np.float32)

        n = len(self.steps)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        return advantages, returns
