"""Tests for JSON-side tactical move selection."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.inference import tactics


def _snake(sid, cells, *, health=90, length=None):
    body = [{"x": x, "y": y} for x, y in cells]
    return {
        "id": sid,
        "health": health,
        "body": body,
        "head": body[0],
        "length": length if length is not None else len(cells),
    }


def _payload(you, others=(), *, food=(), width=15, height=15):
    snakes = [you, *others]
    return {
        "board": {
            "width": width,
            "height": height,
            "food": [{"x": x, "y": y} for x, y in food],
            "hazards": [],
            "snakes": snakes,
        },
        "you": you,
    }


def test_never_moves_into_wall_or_neck() -> None:
    you = _snake("me", [(7, 14), (7, 13), (7, 12)])
    p = _payload(you)
    legal = tactics.legal_moves(p)
    assert "up" not in legal and "down" not in legal
    move, _ = tactics.choose_move(p)
    assert move in ("left", "right")


def test_avoids_head_to_head_with_longer_snake() -> None:
    # Enemy head two cells to the right and longer: stepping right risks a H2H loss.
    you = _snake("me", [(7, 7), (7, 6), (7, 5)])
    enemy = _snake("big", [(9, 7), (10, 7), (11, 7), (12, 7), (13, 7)])
    p = _payload(you, [enemy])
    assert "right" not in tactics.safe_moves(p)


def test_takes_adjacent_food_when_hungry() -> None:
    you = _snake("me", [(7, 7), (7, 6), (7, 5)], health=20)
    p = _payload(you, food=[(8, 7)])
    move, _ = tactics.choose_move(p)
    assert move == "right"


def test_model_only_breaks_ties() -> None:
    # Open board, symmetric left/right — model preference decides.
    you = _snake("me", [(7, 7), (7, 6), (7, 5)])
    p = _payload(you)
    scores = {m: tactics.score_move(p, m) for m in tactics.safe_moves(p)}
    tied = [m for m, s in scores.items() if s >= max(scores.values()) - 0.01]
    if len(tied) > 1:
        move, dbg = tactics.choose_move(p, preferred=tied[-1])
        assert move == tied[-1] and dbg["tie_break_model"]
    # A preference outside the top set is ignored.
    worst = min(scores, key=lambda m: scores[m])
    if scores[worst] < max(scores.values()) - 0.01:
        move, _ = tactics.choose_move(p, preferred=worst)
        assert move != worst


def test_avoids_pocket_smaller_than_body() -> None:
    # Corridor of 2 cells to the left, open board to the right.
    you = _snake("me", [(3, 7), (3, 6), (3, 5), (3, 4), (3, 3)])
    wall = _snake("w", [(1, y) for y in range(3, 12)])
    p = _payload(you, [wall])
    move, _ = tactics.choose_move(p)
    assert move != "left"
