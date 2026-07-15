"""Regression: hisss get_obs must work after mid-game eliminations."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import hisss
from hisss.game.state import BattleSnakeState

from battlesnake_ai.env.hisss_view_radius_fix import apply_view_radius_row_index_fix


def test_view_radius_patch_applies() -> None:
    assert apply_view_radius_row_index_fix() is True


def test_get_obs_after_noncontiguous_alive_players() -> None:
    """Players [1, 3] alive — stock hisss 1.3 indexes obs by pid and crashes."""
    assert apply_view_radius_row_index_fix() is True

    env = hisss.BattleSnakeGame(hisss.restricted_standard_config())
    alive_state = BattleSnakeState(
        turn=0,
        snakes_alive=[True, True, True, True],
        snake_pos={
            0: [(1, 1), (1, 1), (1, 1)],
            1: [(13, 1), (13, 1), (13, 1)],
            2: [(1, 13), (1, 13), (1, 13)],
            3: [(13, 13), (13, 13), (13, 13)],
        },
        food_pos=[[7, 7]],
        snake_health=[100, 100, 100, 100],
        snake_len=[3, 3, 3, 3],
        food_spawn_turns=[0],
        elimination_events=None,
    )
    env.set_state(alive_state)
    obs0, _, _ = env.get_obs()
    assert obs0.shape[0] == 4

    mid_state = BattleSnakeState(
        turn=2,
        snakes_alive=[False, True, False, True],
        snake_pos={
            0: [(0, 0)],
            1: [(13, 3), (13, 2), (13, 1)],
            2: [(0, 0)],
            3: [(1, 3), (1, 2), (1, 1)],
        },
        food_pos=[[7, 7]],
        snake_health=[0, 98, 0, 98],
        snake_len=[3, 3, 3, 3],
        food_spawn_turns=[0],
        elimination_events=None,
    )
    env.set_state(mid_state)
    assert list(env.players_at_turn()) == [1, 3]
    obs, _, _ = env.get_obs()
    assert obs.shape[0] == 2
