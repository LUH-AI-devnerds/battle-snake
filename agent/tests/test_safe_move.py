"""Tests for pure-Python safe move filtering."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.inference.safe_move import choose_safe_move, legal_moves


def _payload(head, neck=None, *, width=15, height=15, food=None):
    body = [head]
    if neck is not None:
        body.append(neck)
    else:
        body.append(head)
    body.append(body[-1])
    you = {
        "id": "me",
        "health": 90,
        "body": [{"x": x, "y": y} for x, y in body],
        "head": {"x": head[0], "y": head[1]},
        "length": 3,
    }
    return {
        "board": {
            "width": width,
            "height": height,
            "food": food or [],
            "hazards": [],
            "snakes": [you],
        },
        "you": you,
    }


def test_rejects_wall_and_reverse() -> None:
    # Head at top edge; up is suicide. Neck below → down is reverse.
    p = _payload((7, 14), neck=(7, 13))
    legal = legal_moves(p)
    assert "up" not in legal
    assert "down" not in legal
    assert "left" in legal and "right" in legal


def test_overrides_model_into_wall() -> None:
    p = _payload((7, 14), neck=(7, 13))
    assert choose_safe_move(p, preferred="up") in ("left", "right")


def test_prefers_space_over_wall_charge() -> None:
    # Near north wall; model prefers up — safe chooser should leave the edge.
    p = _payload((7, 13), neck=(7, 12))
    move = choose_safe_move(p, preferred="up")
    assert move != "up"


def test_fallback_avoids_losing_head_to_head() -> None:
    """The last-resort path must not walk alongside a bigger snake.

    This is what actually served every live game while the model path was
    crashing, and it had no head-to-head logic at all -- the small snake would
    step right next to a much longer one and lose the collision.
    """
    you_body = [(7, 7), (7, 6), (7, 5)]
    you = {
        "id": "me", "health": 90, "length": 3,
        "body": [{"x": x, "y": y} for x, y in you_body],
        "head": {"x": 7, "y": 7},
    }
    big_body = [(9, 7), (10, 7), (11, 7), (12, 7), (13, 7), (14, 7)]
    big = {
        "id": "big", "health": 90, "length": 6,
        "body": [{"x": x, "y": y} for x, y in big_body],
        "head": {"x": 9, "y": 7},
    }
    payload = {
        "board": {"width": 15, "height": 15, "food": [], "hazards": [],
                  "snakes": [you, big]},
        "you": you,
    }
    # Stepping right lands on (8,7), adjacent to the length-6 snake's head.
    assert choose_safe_move(payload) != "right"
