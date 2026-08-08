"""Hostile /move payloads: the decision path must never raise, and must keep
the policy driving.

Two live outages came from payload shape, both invisible from outside because
/move kept answering 200:

  * ``health: null`` on eliminated snakes -> int(None) TypeError -> every move
    after the first elimination served by the crude heuristic
  * the fix for that mapped null to 0, which marked fogged opponents dead ->
    hisss read a sole-survivor board -> is_terminal() -> model skipped entirely

So the bar here is deliberately higher than "does not crash": for any payload
that is playable at all, the RL policy must actually be the thing deciding.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.env.hisss_view_radius_fix import apply_view_radius_row_index_fix

apply_view_radius_row_index_fix()

from battlesnake_ai.inference.runtime import SnakeRuntime

VALID_MOVES = {"up", "down", "left", "right"}
_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "best_checkpoint", "ppo_league_best.pt"
)


@pytest.fixture(scope="module")
def rt():
    os.environ["MOVE_STRATEGY"] = "model"
    runtime = SnakeRuntime(os.path.abspath(_CKPT), device="cpu")
    yield runtime
    runtime.close()


def _snake(sid, cells, *, health=90, length=None, **extra):
    body = [{"x": int(x), "y": int(y)} for x, y in cells]
    s = {
        "id": sid,
        "health": health,
        "body": body,
        "head": body[0] if body else None,
        "length": length if length is not None else len(cells),
    }
    s.update(extra)
    return s


def _payload(you, others=(), *, food=(), w=15, h=15, turn=5, gid="t", **board_extra):
    board = {
        "width": w, "height": h,
        "food": [{"x": int(a), "y": int(b)} for a, b in food],
        "hazards": [],
        "snakes": [you, *others],
    }
    board.update(board_extra)
    return {"game": {"id": gid}, "turn": turn, "board": board, "you": you}


def _me(cells=((7, 7), (7, 6), (7, 5)), **kw):
    return _snake("me", cells, **kw)


def _decide(rt, payload):
    """Every payload must yield a valid move without raising."""
    move = rt.decide_move(payload)
    assert move in VALID_MOVES, f"invalid move {move!r}"
    return move, rt.last_decision()


# ── the two shapes that actually broke production ────────────────────────────


def test_fogged_opponents_null_health_keeps_policy_driving(rt):
    fogged = [
        {"id": f"o{i}", "health": None, "length": None, "body": [], "head": None}
        for i in range(3)
    ]
    p = _payload(_me(health=100), fogged, food=[(3, 3)], gid="fow")
    rt.on_game_start(p)
    _, d = _decide(rt, p)
    assert d["source"].startswith("model_safe"), f"policy not driving: {d['source']}"


def test_dead_snakes_with_null_fields_keep_policy_driving(rt):
    dead = {
        "id": "gone", "health": None, "length": None, "body": [], "head": None,
        "elimination": {"cause": "out-of-health", "turn": 3},
    }
    p = _payload(_me(), [], gid="dead", dead_snakes=[dead])
    rt.on_game_start(p)
    _, d = _decide(rt, p)
    assert d["source"].startswith("model_safe"), f"policy not driving: {d['source']}"


# ── malformed / hostile shapes: must not raise ───────────────────────────────


@pytest.mark.parametrize("mutate,label", [
    (lambda p: p["board"].pop("width"), "missing width"),
    (lambda p: p["board"].pop("height"), "missing height"),
    (lambda p: p["board"].pop("food"), "missing food"),
    (lambda p: p["board"].pop("snakes"), "missing snakes"),
    (lambda p: p.pop("board"), "missing board"),
    (lambda p: p.update(turn=None), "null turn"),
    (lambda p: p["board"].update(food=None), "null food"),
    (lambda p: p["board"].update(snakes=[]), "empty snakes"),
    (lambda p: p["you"].update(body=[]), "empty body"),
    (lambda p: p["you"].update(head=None), "null head"),
    (lambda p: p["you"].update(health=None), "null own health"),
    (lambda p: p["you"].update(length=None), "null own length"),
    (lambda p: p["board"]["food"].append({"x": None, "y": None}), "null food coords"),
    (lambda p: p["board"]["food"].append({"x": -1, "y": -1}), "fow food placeholder"),
    (lambda p: p["you"]["body"].append({"x": None, "y": None}), "null body segment"),
    (lambda p: p["you"]["body"].append({"x": -1, "y": -1}), "fow body placeholder"),
    (lambda p: p["board"]["snakes"].append({"id": "ghost"}), "snake with only an id"),
    (lambda p: p["board"]["snakes"].append({}), "totally empty snake"),
    (lambda p: p.update(game=None), "null game"),
    (lambda p: p.update(game={}), "empty game"),
])
def test_malformed_payloads_never_raise(rt, mutate, label):
    p = _payload(_me(), [_snake("opp", [(2, 2), (2, 1)])], food=[(9, 9)], gid=f"mal-{label}")
    try:
        mutate(p)
    except Exception as exc:  # pragma: no cover - mutation itself must work
        pytest.fail(f"bad mutation {label}: {exc}")
    move = rt.decide_move(p)
    assert move in VALID_MOVES, f"{label}: got {move!r}"


# ── board / game-state edge cases ────────────────────────────────────────────


@pytest.mark.parametrize("w,h", [(7, 7), (11, 11), (15, 15), (19, 19), (25, 25)])
def test_non_default_board_sizes(rt, w, h):
    p = _payload(_me(((w // 2, h // 2), (w // 2, h // 2 - 1))), w=w, h=h, gid=f"b{w}")
    rt.on_game_start(p)
    move, _ = _decide(rt, p)
    assert move in VALID_MOVES


@pytest.mark.parametrize("corner", [(0, 0), (14, 0), (0, 14), (14, 14)])
def test_head_in_every_corner(rt, corner):
    x, y = corner
    ny = y + 1 if y == 0 else y - 1
    p = _payload(_me(((x, y), (x, ny))), gid=f"c{x}{y}")
    rt.on_game_start(p)
    move, _ = _decide(rt, p)
    assert move in VALID_MOVES


def test_sole_survivor(rt):
    p = _payload(_me(), [], gid="solo")
    rt.on_game_start(p)
    move, _ = _decide(rt, p)
    assert move in VALID_MOVES


def test_we_are_already_dead(rt):
    p = _payload(_me(health=0), [_snake("opp", [(2, 2), (2, 1)])], gid="dead-me")
    move, _ = _decide(rt, p)
    assert move in VALID_MOVES


def test_completely_boxed_in_still_answers(rt):
    # Walled into a single cell: no legal move exists, but we must still reply.
    me = _snake("me", [(0, 0)])
    wall = _snake("w", [(0, 1), (1, 1), (1, 0)], length=3)
    p = _payload(me, [wall], gid="boxed")
    move, _ = _decide(rt, p)
    assert move in VALID_MOVES


def test_just_eaten_snake_has_duplicate_tail(rt):
    # After eating, the tail occupies the same cell twice.
    me = _snake("me", [(7, 7), (7, 6), (7, 5), (7, 5)], length=4, health=100)
    p = _payload(me, [_snake("opp", [(2, 2), (2, 1)])], gid="ate")
    rt.on_game_start(p)
    move, _ = _decide(rt, p)
    assert move in VALID_MOVES


def test_very_long_snake(rt):
    cells = [(x, 7) for x in range(14, -1, -1)] + [(0, 8), (1, 8)]
    p = _payload(_snake("me", cells, length=len(cells)), gid="long")
    rt.on_game_start(p)
    move, _ = _decide(rt, p)
    assert move in VALID_MOVES


def test_more_than_four_snakes(rt):
    others = [_snake(f"o{i}", [(i, 0), (i, 1)]) for i in range(5)]
    p = _payload(_me(), others, gid="crowd")
    rt.on_game_start(p)
    move, _ = _decide(rt, p)
    assert move in VALID_MOVES


def test_huge_turn_number(rt):
    p = _payload(_me(), turn=10**9, gid="bigturn")
    rt.on_game_start(p)
    move, _ = _decide(rt, p)
    assert move in VALID_MOVES


def test_unknown_extra_fields_are_ignored(rt):
    me = _me(shout="hello", squad="a", latency="12", customizations={"color": "#fff"})
    p = _payload(me, gid="extra", ruleset={"name": "royale"}, someNewField=[1, 2, 3])
    rt.on_game_start(p)
    move, _ = _decide(rt, p)
    assert move in VALID_MOVES


def test_game_id_switch_midstream_resets_state(rt):
    a = _payload(_me(), [_snake("opp", [(2, 2), (2, 1)])], gid="game-a")
    rt.on_game_start(a)
    _decide(rt, a)
    # A different game id must not inherit the previous game's snake mapping.
    b = _payload(_snake("other", [(4, 4), (4, 3)]), [_snake("x", [(9, 9), (9, 8)])], gid="game-b")
    move, d = _decide(rt, b)
    assert move in VALID_MOVES
    assert d["source"].startswith("model_safe"), f"policy not driving after switch: {d['source']}"


def test_full_game_sequence_keeps_policy_driving(rt):
    """A whole game start->moves->end, asserting the policy drives throughout."""
    gid = "seq"
    body = [(7, 7), (7, 6), (7, 5)]
    p = _payload(_snake("me", body), [_snake("o", [(2, 2), (2, 1)])], food=[(9, 9)], gid=gid)
    rt.on_game_start(p)
    sources = []
    for turn in range(1, 25):
        p = _payload(_snake("me", body), [_snake("o", [(2, 2), (2, 1)])],
                     food=[(9, 9)], turn=turn, gid=gid)
        move, d = _decide(rt, p)
        sources.append(d["source"])
        dx, dy = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}[move]
        nxt = (body[0][0] + dx, body[0][1] + dy)
        if not (0 <= nxt[0] < 15 and 0 <= nxt[1] < 15):
            break
        body = [nxt] + body[:-1]
    rt.on_game_end(p)
    model_driven = sum(1 for s in sources if s.startswith("model_safe"))
    assert model_driven == len(sources), f"policy stopped driving: {set(sources)}"


# ── the stronger bar: for a *playable* payload the policy must actually drive ──

@pytest.mark.parametrize("mutate,label", [
    (lambda p: p["board"]["food"].append({"x": None, "y": None}), "null food coords"),
    (lambda p: (p["board"].pop("width"), p["board"].pop("height")), "missing dimensions"),
    (lambda p: p["board"]["snakes"].append(
        {"id": "fog", "health": None, "length": None, "body": [], "head": None}), "fogged opponent"),
    (lambda p: p["board"]["snakes"].append(
        {"id": "bad", "health": 80, "length": 2,
         "body": [{"x": None, "y": None}, {"x": 2, "y": 1}], "head": {"x": 2, "y": 2}}),
     "null segment on opponent"),
    (lambda p: p["board"].update(dead_snakes=[
        {"id": "gone", "health": None, "length": None, "body": [], "head": None,
         "elimination": {"cause": "wall-collision", "turn": 2}}]), "dead_snakes null fields"),
])
def test_playable_payloads_keep_the_policy_driving(rt, mutate, label):
    """Not crashing is not enough -- falling back to the heuristic is a ~0% win
    rate policy, which is how two live outages stayed invisible. Anything the
    engine can actually be played from must be decided by the model."""
    p = _payload(_me(), [_snake("opp", [(2, 2), (2, 1)])], food=[(9, 9)], gid=f"drive-{label}")
    mutate(p)
    rt.on_game_start(p)
    move, d = _decide(rt, p)
    assert d["source"].startswith("model_safe"), f"{label}: fell back to {d['source']}"


def test_default_strategy_matches_the_deployed_one():
    """The code default must equal what the Dockerfile ships.

    These drifted apart once: the code defaulted to "veto" while the image
    pinned "model", so a host without the env ran an unvalidated configuration.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "Dockerfile").read_text()
    m = re.search(r'ENV MOVE_STRATEGY="([a-z]+)"', dockerfile)
    assert m, "Dockerfile must pin MOVE_STRATEGY explicitly"
    shipped = m.group(1)

    runtime_src = (root / "agent/src/battlesnake_ai/inference/runtime.py").read_text()
    d = re.search(r'os\.environ\.get\("MOVE_STRATEGY",\s*"([a-z]+)"\)', runtime_src)
    assert d, "runtime must have an explicit MOVE_STRATEGY default"
    assert d.group(1) == shipped, (
        f"code default {d.group(1)!r} != Dockerfile {shipped!r}"
    )
