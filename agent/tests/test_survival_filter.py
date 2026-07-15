"""Unit tests for grow-then-hunt / survival action filtering."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.inference.survival import (
    filter_survival_actions,
    is_losing_head_to_head,
    is_winning_head_to_head,
    next_cell,
    rank_survival_actions,
    select_survival_action,
)


def test_h2h_detection() -> None:
    enemies = [(5, 6, 5)]  # longer
    assert is_losing_head_to_head((5, 6), our_len=3, enemies=enemies)
    assert is_losing_head_to_head((6, 6), our_len=3, enemies=enemies)
    assert not is_losing_head_to_head((6, 6), our_len=6, enemies=enemies)
    prey = [(5, 6, 2)]  # shorter
    assert is_winning_head_to_head((5, 6), our_len=5, enemies=prey)
    assert is_winning_head_to_head((6, 6), our_len=5, enemies=prey)
    assert not is_winning_head_to_head((6, 6), our_len=2, enemies=prey)


def test_next_cell() -> None:
    assert next_cell((3, 3), 0) == (3, 4)
    assert next_cell((3, 3), 1) == (4, 3)
    assert next_cell((3, 3), 2) == (3, 2)
    assert next_cell((3, 3), 3) == (2, 3)


class _FakeEnvLongerThreat:
    """Us length 3; longer enemy head at (5,6)."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(w=15, h=15)
        self._state = SimpleNamespace(
            snakes_alive=[True, True],
            snake_pos={
                0: [(5, 5), (5, 4), (5, 3)],
                1: [(5, 6), (5, 7), (5, 8), (5, 9), (5, 10)],
            },
            snake_len=[3, 5],
            snake_health=[80, 80],
            food_pos=[[6, 5]],
        )

    def available_actions(self, pid: int):
        return [0, 1, 2, 3]

    def get_state(self):
        return self._state


class _FakeEnvHunter:
    """Us length 6; shorter prey head at (7,5). Gap cell (6,5)=right is a hunt."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(w=15, h=15)
        self._state = SimpleNamespace(
            snakes_alive=[True, True],
            snake_pos={
                0: [(5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0)],
                1: [(7, 5), (8, 5), (9, 5)],
            },
            snake_len=[6, 3],
            snake_health=[80, 80],
            food_pos=[[1, 1]],
        )

    def available_actions(self, pid: int):
        return [0, 1, 2, 3]

    def get_state(self):
        return self._state


class _FakeEnvNeedFood:
    """Us length 3 vs enemy length 5 — need to grow; food to the right."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(w=15, h=15)
        self._state = SimpleNamespace(
            snakes_alive=[True, True],
            snake_pos={
                0: [(5, 5), (4, 5), (3, 5)],
                1: [(12, 12), (12, 11), (12, 10), (12, 9), (12, 8)],
            },
            snake_len=[3, 5],
            snake_health=[70, 80],
            food_pos=[[6, 5]],
        )

    def available_actions(self, pid: int):
        return [0, 1, 2, 3]

    def get_state(self):
        return self._state


def test_filters_losing_h2h() -> None:
    env = _FakeEnvLongerThreat()
    safe = filter_survival_actions(env, 0)
    assert 0 not in safe
    assert 1 in safe
    assert 3 in safe


def test_hunts_shorter_prey() -> None:
    env = _FakeEnvHunter()
    q = np.zeros(4, dtype=np.float32)
    ranked = rank_survival_actions(env, 0, q_values=q, strategy="aggressive")
    # Right contests the shorter prey at (7,5)
    assert ranked[0] == 1


def test_seeks_food_when_not_longest() -> None:
    env = _FakeEnvNeedFood()
    q = np.zeros(4, dtype=np.float32)
    ranked = rank_survival_actions(env, 0, q_values=q, strategy="aggressive")
    assert ranked[0] == 1  # food at (6,5)


def test_select_uses_q_when_dominant() -> None:
    env = _FakeEnvLongerThreat()
    q = np.array([-10.0, 5.0, -10.0, -10.0], dtype=np.float32)
    a = select_survival_action(env, 0, q, strategy="aggressive")
    assert a == 1


if __name__ == "__main__":
    test_h2h_detection()
    test_next_cell()
    test_filters_losing_h2h()
    test_hunts_shorter_prey()
    test_seeks_food_when_not_longest()
    test_select_uses_q_when_dominant()
    print("ok")
