"""On-policy rollout storage for PPO.

Transitions are stored per snake seat.  A 4-player game interleaves four
independent trajectories in time, so GAE must run per seat — bootstrapping
across seats mixes unrelated value estimates and destroys the advantage signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RolloutStep:
    obs: np.ndarray
    action: int
    log_prob: float
    reward: float
    done: bool
    value: float
    # Legal-action mask used when the action was sampled. The update must apply
    # the same mask or the importance ratio compares two different distributions.
    legal_mask: Optional[np.ndarray] = None


class RolloutBuffer:
    """A single trajectory (one seat, contiguous in time)."""

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


@dataclass
class RolloutBatch:
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    legal_masks: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.shape[0])


class MultiSeatRollout:
    """Per-seat trajectories collected from one multi-snake environment."""

    def __init__(self, num_actions: int = 4) -> None:
        self.num_actions = num_actions
        self.seats: Dict[int, RolloutBuffer] = {}

    def push(self, seat: int, step: RolloutStep) -> None:
        self.seats.setdefault(seat, RolloutBuffer()).push(step)

    def mark_done(self, seat: int) -> None:
        """Terminate the seat's most recent step (snake died / episode ended)."""
        buf = self.seats.get(seat)
        if buf and buf.steps:
            buf.steps[-1].done = True

    def clear(self) -> None:
        self.seats.clear()

    def __len__(self) -> int:
        return sum(len(b) for b in self.seats.values())

    def build(
        self,
        last_values: Dict[int, float],
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        obs_list: List[np.ndarray] = []
        actions: List[int] = []
        log_probs: List[float] = []
        values: List[float] = []
        adv_list: List[np.ndarray] = []
        ret_list: List[np.ndarray] = []
        masks: List[np.ndarray] = []

        for seat, buf in self.seats.items():
            if not len(buf):
                continue
            adv, ret = buf.compute_gae(float(last_values.get(seat, 0.0)), gamma, gae_lambda)
            adv_list.append(adv)
            ret_list.append(ret)
            for s in buf.steps:
                obs_list.append(s.obs)
                actions.append(s.action)
                log_probs.append(s.log_prob)
                values.append(s.value)
                mask = s.legal_mask
                if mask is None:
                    mask = np.ones(self.num_actions, dtype=bool)
                masks.append(np.asarray(mask, dtype=bool))

        return RolloutBatch(
            obs=np.stack(obs_list, axis=0),
            actions=np.asarray(actions, dtype=np.int64),
            log_probs=np.asarray(log_probs, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
            advantages=np.concatenate(adv_list) if adv_list else np.zeros(0, dtype=np.float32),
            returns=np.concatenate(ret_list) if ret_list else np.zeros(0, dtype=np.float32),
            legal_masks=np.stack(masks, axis=0),
        )
