"""API adapter edge cases for Blackout /move payloads."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.inference.api_adapter import assign_player_ids, request_to_state


def _payload(*, turn=0) -> dict:
    body = [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}]
    snake = {
        "id": "me",
        "name": "42",
        "health": 100,
        "body": body,
        "head": body[0],
        "length": 3,
        "latency": "1",
        "shout": "",
        "squad": "",
        "customizations": {"color": "#00f", "head": "default", "tail": "default"},
    }
    return {
        "game": {"id": "g1"},
        "turn": turn,
        "board": {
            "width": 15,
            "height": 15,
            "food": [{"x": 7, "y": 7, "spawn_turn": 0}],
            "hazards": [],
            "snakes": [snake],
        },
        "you": snake,
    }


def test_request_to_state_accepts_null_turn() -> None:
    payload = _payload(turn=None)
    pids = assign_player_ids(payload)
    state, your_pid = request_to_state(payload, pid_by_snake_id=pids)
    assert state.turn == 0
    assert your_pid == 0


def test_request_to_state_accepts_null_fields_on_dead_snakes() -> None:
    """Regression: the live Blackout API reports health/length as JSON null
    (not absent) for eliminated snakes in dead_snakes. dict.get(key, default)
    only substitutes default when the key is missing, so this used to raise
    TypeError: int() argument ... not 'NoneType' on every /move once a game
    had an elimination -- falling back to the crude safe-move heuristic for
    the rest of the match instead of running the trained policy.
    """
    you_body = [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}]
    you = {"id": "me", "health": 90, "body": you_body, "head": you_body[0], "length": 3}
    dead_snake = {
        "id": "opp",
        "health": None,
        "body": [],
        "head": None,
        "length": None,
        "elimination": {"cause": "out-of-health", "turn": 12},
    }
    payload = {
        "game": {"id": "g1"},
        "turn": 13,
        "board": {
            "width": 15,
            "height": 15,
            "food": [{"x": 7, "y": 7, "spawn_turn": None}],
            "hazards": [],
            "snakes": [you],
            "dead_snakes": [dead_snake],
        },
        "you": you,
    }
    # In production the id->pid map is persisted across turns (SnakeRuntime.
    # _pid_by_snake_id), so an opponent seen alive earlier is still mapped
    # once it moves to dead_snakes and drops out of board.snakes.
    pids = {"me": 0, "opp": 1}
    state, your_pid = request_to_state(payload, pid_by_snake_id=pids)
    assert your_pid == pids["me"]
    opp_pid = pids["opp"]
    assert state.snake_health[opp_pid] == 0
    assert state.snakes_alive[opp_pid] is False
