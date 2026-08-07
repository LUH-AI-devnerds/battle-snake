"""Live-match telemetry.

The competition server offers no shell while a leaderboard run is happening,
so these counters are the only way to tell whether the policy is actually
playing. A fallback must never be reportable as healthy.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.inference.telemetry import Telemetry


def _decision(turn, source="model_safe/model", ms=12.0, move="up"):
    return {"move": move, "source": source, "ms": ms, "turn": turn, "legal": ["up", "left"]}


def _you(length=5, health=90):
    return {"id": "me", "length": length, "health": health}


def test_counts_moves_and_reports_healthy():
    t = Telemetry()
    t.on_start("g1")
    for turn in range(5):
        t.on_move("g1", _decision(turn), _you(length=3 + turn))
    snap = t.snapshot()
    assert snap["games"] == 1
    assert snap["moves"] == 5
    assert snap["fallback_moves"] == 0
    assert snap["fallback_rate"] == 0.0
    assert t.healthy()
    assert snap["move_sources"]["model_safe/model"] == 5


def test_fallback_is_counted_and_makes_it_unhealthy():
    t = Telemetry()
    t.on_start("g1")
    t.on_move("g1", _decision(0), _you())
    t.on_move("g1", _decision(1, source="safe_exception"), _you())
    snap = t.snapshot()
    assert snap["fallback_moves"] == 1
    assert snap["fallback_rate"] == 0.5
    assert snap["games_with_fallback"] == 1
    assert not t.healthy(), "a fallback must never report healthy"


def test_over_budget_moves_are_flagged():
    t = Telemetry()
    t.on_start("g1")
    t.on_move("g1", _decision(0, ms=15.0), _you())
    t.on_move("g1", _decision(1, ms=780.0), _you())
    snap = t.snapshot()
    assert snap["over_budget_moves"] == 1
    assert not t.healthy()


def test_end_classifies_outcomes():
    t = Telemetry()

    t.on_start("won")
    t.on_move("won", _decision(0), _you(length=12))
    me = {"id": "me", "health": 80, "length": 12}
    summary = t.on_end("won", {"you": me, "board": {"snakes": [me]}})
    assert summary["outcome"] == "won"

    t.on_start("dead")
    t.on_move("dead", _decision(0), _you(length=4))
    dead_me = {"id": "me", "health": 0, "length": 4}
    other = {"id": "opp", "health": 70, "length": 9}
    summary = t.on_end("dead", {"you": dead_me, "board": {"snakes": [dead_me, other]}})
    assert summary["outcome"] == "eliminated"

    assert t.snapshot()["outcomes"] == {"won": 1, "eliminated": 1}


def test_ring_buffers_stay_bounded():
    t = Telemetry(max_games=3, max_decisions=5)
    for g in range(6):
        gid = f"g{g}"
        t.on_start(gid)
        for turn in range(4):
            t.on_move(gid, _decision(turn), _you())
    snap = t.snapshot(recent_games=50)
    assert len(snap["recent_games"]) <= 3
    assert len(t.recent_decisions(100)) <= 5
    # Totals survive eviction even though per-game records do not.
    assert snap["moves"] == 24
    assert snap["games"] == 6


def test_recent_decisions_are_newest_first():
    t = Telemetry()
    t.on_start("g1")
    for turn in range(3):
        t.on_move("g1", _decision(turn, move="up" if turn else "left"), _you())
    recent = t.recent_decisions(3)
    assert [d["turn"] for d in recent] == [2, 1, 0]


def test_handler_exception_counts_as_a_fallback():
    """A crash in the /move handler itself must not read as healthy.

    A malformed board once made /move return 500 -- a forfeited move -- while
    telemetry recorded nothing at all, so /stats reported healthy during a
    total failure. The handler now catches, returns a legal move, and records
    the failure under this source.
    """
    t = Telemetry()
    t.on_start("g1")
    t.on_move("g1", _decision(0, source="handler_exception"), _you())
    snap = t.snapshot()
    assert snap["fallback_moves"] == 1
    assert not t.healthy()
