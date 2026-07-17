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
