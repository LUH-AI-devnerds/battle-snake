"""Joint action selection for multi-snake hisss environments."""

from __future__ import annotations

import random
from typing import Any, Callable, List, Optional, Set, Tuple

import numpy as np
import torch


def joint_legal_set(env: Any) -> Set[Tuple[int, ...]]:
    return set(tuple(x) for x in env.available_joint_actions())


def masked_argmax(q: np.ndarray, legal: List[int]) -> int:
    best = legal[0]
    best_v = q[best]
    for a in legal[1:]:
        if q[a] > best_v:
            best_v = q[a]
            best = a
    return int(best)


def masked_softmax_sample(
    logits: np.ndarray,
    legal: List[int],
    temperature: float = 1.0,
) -> int:
    if len(legal) == 1:
        return int(legal[0])
    sub = logits[legal] / max(temperature, 1e-8)
    sub = sub - sub.max()
    exp = np.exp(sub)
    probs = exp / exp.sum()
    return int(np.random.choice(legal, p=probs))


def epsilon_greedy_joint(
    env: Any,
    obs: np.ndarray,
    epsilon: float,
    q_fn: Callable[[np.ndarray], np.ndarray],
) -> Tuple[int, ...]:
    """
    Independent ε-greedy per alive snake on masked Q-values; if the greedy tuple is not
    jointly legal, sample a uniform random legal joint action.
    """
    joint_set = joint_legal_set(env)
    players_here = list(env.players_at_turn())

    if random.random() < epsilon:
        return tuple(random.choice(env.available_joint_actions()))

    greedy: List[int] = []
    for row_idx, pid in enumerate(players_here):
        q = q_fn(obs[row_idx : row_idx + 1])
        la = env.available_actions(pid)
        greedy.append(masked_argmax(q, la))
    tup = tuple(greedy)
    if tup in joint_set:
        return tup
    return tuple(random.choice(env.available_joint_actions()))


def select_joint_actions_epsilon_greedy(
    env: Any,
    policy: Any,
    obs: np.ndarray,
    epsilon: float,
) -> Tuple[int, ...]:
    """ε-greedy joint actions using a model with ``forward(obs_np) -> Q``."""

    def q_fn(slice_obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return policy(slice_obs).detach().cpu().numpy()[0]

    return epsilon_greedy_joint(env, obs, epsilon, q_fn)


def stochastic_joint(
    env: Any,
    obs: np.ndarray,
    logits_fn: Callable[[np.ndarray], np.ndarray],
    temperature: float = 1.0,
) -> Tuple[int, ...]:
    """Sample per-snake actions from masked softmax; fall back to random joint if illegal."""
    joint_set = joint_legal_set(env)
    players_here = list(env.players_at_turn())
    sampled: List[int] = []
    for row_idx, pid in enumerate(players_here):
        logits = logits_fn(obs[row_idx : row_idx + 1])
        la = env.available_actions(pid)
        sampled.append(masked_softmax_sample(logits, la, temperature))
    tup = tuple(sampled)
    if tup in joint_set:
        return tup
    return tuple(random.choice(env.available_joint_actions()))


def ensemble_joint(
    env: Any,
    obs: np.ndarray,
    score_fns: List[Tuple[float, Callable[[np.ndarray], np.ndarray]]],
) -> Tuple[int, ...]:
    """
    Weighted sum of per-action scores from multiple policies, then masked argmax per snake.
    Each score_fn returns a 1d array of length num_actions.
    """
    joint_set = joint_legal_set(env)
    players_here = list(env.players_at_turn())
    greedy: List[int] = []
    for row_idx, pid in enumerate(players_here):
        sl = obs[row_idx : row_idx + 1]
        combined = None
        for weight, fn in score_fns:
            scores = fn(sl) * weight
            combined = scores if combined is None else combined + scores
        assert combined is not None
        la = env.available_actions(pid)
        greedy.append(masked_argmax(combined, la))
    tup = tuple(greedy)
    if tup in joint_set:
        return tup
    return tuple(random.choice(env.available_joint_actions()))


def random_joint(env: Any) -> Tuple[int, ...]:
    return tuple(random.choice(env.available_joint_actions()))
