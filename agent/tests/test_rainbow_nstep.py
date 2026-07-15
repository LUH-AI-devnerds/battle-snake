"""Unit tests for n-step return accumulation in RainbowTrainingLoop.

Verifies that the per-snake n-step buffer produces correctly accumulated
n-step returns and gamma_n exponents, including the terminal flush of
partial windows at episode end / snake death.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.models.rainbow_dqn import RainbowDQN
from battlesnake_ai.training.prioritized_replay import PrioritizedReplayBuffer
from battlesnake_ai.training.rainbow_loop import RainbowTrainingLoop


def _make_loop(gamma: float, n_step: int, capacity: int = 100) -> RainbowTrainingLoop:
    replay = PrioritizedReplayBuffer(capacity=capacity, alpha=0.6)
    policy = RainbowDQN(in_channels=1, num_atoms=5, feature_dim=4)
    target = RainbowDQN(in_channels=1, num_atoms=5, feature_dim=4)
    loop = RainbowTrainingLoop(
        env=None,
        policy_net=policy,
        target_net=target,
        replay=replay,
        metrics=None,
        gamma=gamma,
        n_step=n_step,
        device=torch.device("cpu"),
        eval_every=0,
    )
    loop._nstep_buf = {}
    return loop


def _obs(tag: int) -> np.ndarray:
    # Tiny unique observation so we can identify the source step.
    return np.full((2, 2, 1), float(tag), dtype=np.float32)


def _transitions(loop: RainbowTrainingLoop):
    data = loop.replay.tree.data
    n = loop.replay.tree.n_entries
    return [data[i] for i in range(n)]


def test_nstep_accumulation_and_flush() -> None:
    gamma = 0.99
    n = 3
    loop = _make_loop(gamma=gamma, n_step=n)

    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, r in enumerate(rewards):
        loop._nstep_append(0, _obs(i), action=i, reward=r)
        loop._nstep_emit_ready(0)
    # Two full windows have been emitted (oldest two entries), three remain buffered.
    loop._nstep_flush(0)

    ts = _transitions(loop)
    assert len(ts) == 5, f"expected 5 transitions, got {len(ts)}"

    # Non-terminal windows: (r0..r2) and (r1..r3)
    assert ts[0].done is False
    assert ts[0].action == 0
    expected_0 = rewards[0] + gamma * rewards[1] + gamma**2 * rewards[2]
    assert abs(ts[0].n_step_return - expected_0) < 1e-5
    assert abs(ts[0].gamma_n - gamma**n) < 1e-7
    # next_obs for the first window is the newest entry's obs at emit time (obs index 3).
    assert float(ts[0].next_obs[0, 0, 0]) == 3.0

    assert ts[1].done is False
    assert ts[1].action == 1
    expected_1 = rewards[1] + gamma * rewards[2] + gamma**2 * rewards[3]
    assert abs(ts[1].n_step_return - expected_1) < 1e-5
    assert abs(ts[1].gamma_n - gamma**n) < 1e-7
    assert float(ts[1].next_obs[0, 0, 0]) == 4.0

    # Terminal flush of remaining three entries: k = 3, 2, 1
    assert ts[2].done is True
    assert ts[2].action == 2
    expected_2 = rewards[2] + gamma * rewards[3] + gamma**2 * rewards[4]
    assert abs(ts[2].n_step_return - expected_2) < 1e-5
    assert abs(ts[2].gamma_n - gamma**3) < 1e-7

    assert ts[3].done is True
    assert ts[3].action == 3
    expected_3 = rewards[3] + gamma * rewards[4]
    assert abs(ts[3].n_step_return - expected_3) < 1e-5
    assert abs(ts[3].gamma_n - gamma**2) < 1e-7

    assert ts[4].done is True
    assert ts[4].action == 4
    assert abs(ts[4].n_step_return - rewards[4]) < 1e-5
    assert abs(ts[4].gamma_n - gamma) < 1e-7


def test_nstep_one_step_matches_q_learning() -> None:
    """With n_step=1, n-step reduces to ordinary 1-step TD targets."""
    gamma = 0.99
    loop = _make_loop(gamma=gamma, n_step=1)

    loop._nstep_append(0, _obs(0), action=0, reward=1.0)
    loop._nstep_emit_ready(0)  # len=2 > 1 -> emit oldest
    loop._nstep_append(0, _obs(1), action=1, reward=2.0)
    loop._nstep_emit_ready(0)
    loop._nstep_flush(0)

    ts = _transitions(loop)
    assert len(ts) == 2
    assert ts[0].done is False
    assert abs(ts[0].n_step_return - 1.0) < 1e-5
    assert abs(ts[0].gamma_n - gamma) < 1e-7
    assert ts[1].done is True
    assert abs(ts[1].n_step_return - 2.0) < 1e-5
    assert abs(ts[1].gamma_n - gamma) < 1e-7


def test_per_snake_buffers_are_independent() -> None:
    gamma = 0.99
    loop = _make_loop(gamma=gamma, n_step=3)

    # Interleave two snakes; their buffers must not cross-contaminate.
    loop._nstep_append(0, _obs(0), action=0, reward=1.0)
    loop._nstep_emit_ready(0)
    loop._nstep_append(1, _obs(10), action=5, reward=10.0)
    loop._nstep_emit_ready(1)

    loop._nstep_flush(0)
    loop._nstep_flush(1)

    ts = _transitions(loop)
    # Each snake flushed a single-entry window as terminal.
    assert len(ts) == 2
    actions = sorted(t.action for t in ts)
    assert actions == [0, 5]
    rewards = sorted(t.n_step_return for t in ts)
    assert rewards == [1.0, 10.0]


if __name__ == "__main__":
    test_nstep_accumulation_and_flush()
    test_nstep_one_step_matches_q_learning()
    test_per_snake_buffers_are_independent()
    print("ok")
