"""FOW: hidden opponents must not make the hisss env terminal."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.inference.api_adapter import assign_player_ids, request_to_state


def test_fow_hidden_opponents_keep_game_nonterminal() -> None:
    """Only our snake visible — others must stay alive as ghosts."""
    you = {
        "id": "me",
        "name": "42",
        "health": 100,
        "body": [{"x": 1, "y": 1}, {"x": 1, "y": 1}, {"x": 1, "y": 1}],
        "head": {"x": 1, "y": 1},
        "length": 3,
    }
    payload = {
        "game": {"id": "g"},
        "turn": 0,
        "board": {
            "width": 15,
            "height": 15,
            "food": [],
            "hazards": [],
            "snakes": [you],  # FOW: only us
        },
        "you": you,
    }
    # Pretend /start saw 4 snakes.
    pid_map = {
        "me": 0,
        "a": 1,
        "b": 2,
        "c": 3,
    }
    state, your_pid = request_to_state(payload, pid_by_snake_id=pid_map)
    assert your_pid == 0
    assert sum(1 for a in state.snakes_alive if a) >= 2
    assert state.snakes_alive[0] is True

    env = make_env("restricted_standard", num_players=4)
    env.set_state(state)
    assert not env.is_terminal()
    obs, _, _ = env.get_obs()
    assert obs.shape[0] >= 1
    env.close()
