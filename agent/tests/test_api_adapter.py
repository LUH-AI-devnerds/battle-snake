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


def test_fog_of_war_null_health_opponents_stay_alive() -> None:
    """Under fog of war only our own health is visible; opponents report null.

    Treating that as health 0 marked every unseen snake dead, so hisss read
    the board as a sole-survivor win, is_terminal() went true, and the model
    was skipped on every turn of a real match -- /move still answered 200 from
    the crude heuristic, so nothing looked broken from outside.
    """
    me_body = [{"x": 3, "y": 3}, {"x": 3, "y": 2}, {"x": 3, "y": 1}]
    me = {"id": "me", "health": 100, "body": me_body, "head": me_body[0], "length": 3}
    # Opponents present on the board but fogged: health/length null, no body.
    fogged = [
        {"id": f"opp{i}", "health": None, "length": None, "body": [], "head": None}
        for i in range(3)
    ]
    payload = {
        "game": {"id": "fow"},
        "turn": 0,
        "board": {"width": 15, "height": 15, "food": [], "hazards": [],
                  "snakes": [me, *fogged]},
        "you": me,
    }
    pids = assign_player_ids(payload)
    state, your_pid = request_to_state(payload, pid_by_snake_id=pids)

    assert all(state.snakes_alive), f"fogged opponents must stay alive: {state.snakes_alive}"
    assert state.snakes_alive[your_pid] is True
    assert sum(1 for a in state.snakes_alive if a) == 4


def test_explicitly_eliminated_snake_is_still_retired() -> None:
    """The null-health relaxation must not resurrect genuinely dead snakes."""
    me_body = [{"x": 3, "y": 3}, {"x": 3, "y": 2}]
    me = {"id": "me", "health": 90, "body": me_body, "head": me_body[0], "length": 2}
    dead = {"id": "opp", "health": None, "length": None, "body": [], "head": None,
            "elimination": {"cause": "wall-collision", "turn": 4}}
    payload = {
        "game": {"id": "g"},
        "turn": 5,
        "board": {"width": 15, "height": 15, "food": [], "hazards": [],
                  "snakes": [me], "dead_snakes": [dead]},
        "you": me,
    }
    pids = {"me": 0, "opp": 1}
    state, _ = request_to_state(payload, pid_by_snake_id=pids)
    assert state.snakes_alive[1] is False
    assert state.snake_health[1] == 0
