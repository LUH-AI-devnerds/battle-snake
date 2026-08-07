"""In-process telemetry for live Blackout matches.

Built for the case where the only access to a running competition server is
HTTP: no shell, no log tailing while a match is in progress. A null-health
TypeError once put the RL agent out of action for thousands of live moves
while every response stayed 200 OK, so the goal here is that any repeat of
that is visible from a single ``GET /stats``.

Everything is bounded and in-memory -- the process is the only consumer, and
a competition server should never trade move latency for bookkeeping.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from threading import Lock
from typing import Any, Deque, Dict, List, Mapping, Optional

# Blackout allows 500 ms per move. Warn well before that so a slow deploy is
# caught while it is merely slow rather than after it forfeits moves.
LATENCY_WARN_MS = 250.0
LATENCY_BUDGET_MS = 500.0


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 1)


class GameRecord:
    """Per-match rollup. One of these per game id, capped by the ring buffer."""

    __slots__ = (
        "game_id", "started_at", "ended_at", "turns", "fallbacks", "sources",
        "latencies", "last_turn", "our_length", "our_health", "alive_at_end",
        "snakes_at_end", "over_budget", "outcome",
    )

    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        self.started_at = time.time()
        self.ended_at: Optional[float] = None
        self.turns = 0
        self.fallbacks = 0
        self.sources: Counter = Counter()
        self.latencies: List[float] = []
        self.last_turn = 0
        self.our_length = 0
        self.our_health = 0
        self.alive_at_end: Optional[bool] = None
        self.snakes_at_end: Optional[int] = None
        self.over_budget = 0
        self.outcome: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "turns": self.turns,
            "last_turn": self.last_turn,
            "outcome": self.outcome or ("in_progress" if self.ended_at is None else "unknown"),
            "our_length": self.our_length,
            "our_health": self.our_health,
            "snakes_at_end": self.snakes_at_end,
            "fallbacks": self.fallbacks,
            "over_budget": self.over_budget,
            "sources": dict(self.sources),
            "latency_ms": {
                "p50": _percentile(self.latencies, 50),
                "p95": _percentile(self.latencies, 95),
                "max": round(max(self.latencies), 1) if self.latencies else 0.0,
            },
            "duration_s": round((self.ended_at or time.time()) - self.started_at, 1),
        }


class Telemetry:
    """Bounded, thread-safe request telemetry.

    Uvicorn runs the sync route handlers in a worker thread pool, so several
    /move calls can land concurrently; every mutation here takes a lock.
    """

    def __init__(self, *, max_games: int = 60, max_decisions: int = 300) -> None:
        self._lock = Lock()
        self._games: Dict[str, GameRecord] = {}
        self._order: Deque[str] = deque(maxlen=max_games)
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=max_decisions)
        self._totals = Counter()
        self._latencies: Deque[float] = deque(maxlen=2000)
        self._started_at = time.time()

    # ── recording ────────────────────────────────────────────────────────────

    def _record_for(self, game_id: str) -> GameRecord:
        rec = self._games.get(game_id)
        if rec is None:
            rec = GameRecord(game_id)
            self._games[game_id] = rec
            if len(self._order) == self._order.maxlen:
                evicted = self._order[0]
                self._games.pop(evicted, None)
            self._order.append(game_id)
            self._totals["games"] += 1
        return rec

    def on_start(self, game_id: str) -> None:
        with self._lock:
            self._record_for(game_id)

    def on_move(self, game_id: str, decision: Mapping[str, Any], you: Mapping[str, Any]) -> None:
        source = str(decision.get("source") or "unknown")
        ms = float(decision.get("ms") or 0.0)
        is_fallback = "exception" in source or source == "safe"
        with self._lock:
            rec = self._record_for(game_id)
            rec.turns += 1
            rec.last_turn = int(decision.get("turn") or 0)
            rec.sources[source] += 1
            rec.latencies.append(ms)
            rec.our_length = int(you.get("length") or rec.our_length)
            rec.our_health = int(you.get("health") or 0)
            if is_fallback:
                rec.fallbacks += 1
                self._totals["fallback_moves"] += 1
            if ms > LATENCY_BUDGET_MS:
                rec.over_budget += 1
                self._totals["over_budget"] += 1
            elif ms > LATENCY_WARN_MS:
                self._totals["slow_moves"] += 1

            self._totals["moves"] += 1
            self._totals[f"source:{source}"] += 1
            self._latencies.append(ms)
            self._recent.append({
                "game_id": game_id,
                "turn": rec.last_turn,
                "move": decision.get("move"),
                "source": source,
                "ms": round(ms, 1),
                "length": rec.our_length,
                "health": rec.our_health,
                "legal": decision.get("legal"),
            })

    def on_end(self, game_id: str, payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Close out a game and classify the outcome from the final board."""
        you = payload.get("you") or {}
        board = payload.get("board") or {}
        snakes = list(board.get("snakes") or [])
        you_id = str(you.get("id", ""))

        def _alive(s: Mapping[str, Any]) -> bool:
            if s.get("elimination") or s.get("elimination_event"):
                return False
            return int(s.get("health") or 0) > 0

        living = [s for s in snakes if _alive(s)]
        we_alive = any(str(s.get("id", "")) == you_id for s in living)

        with self._lock:
            rec = self._record_for(game_id)
            rec.ended_at = time.time()
            rec.alive_at_end = we_alive
            rec.snakes_at_end = len(living)
            rec.our_length = int(you.get("length") or rec.our_length)
            if we_alive and len(living) <= 1:
                rec.outcome = "won"
            elif we_alive:
                rec.outcome = "survived"
            else:
                rec.outcome = "eliminated"
            self._totals[f"outcome:{rec.outcome}"] += 1
            return rec.summary()

    # ── reporting ────────────────────────────────────────────────────────────

    def snapshot(self, *, recent_games: int = 10) -> Dict[str, Any]:
        with self._lock:
            moves = self._totals["moves"]
            fallback_moves = self._totals["fallback_moves"]
            lat = list(self._latencies)
            sources = {
                k.split(":", 1)[1]: v for k, v in self._totals.items() if k.startswith("source:")
            }
            outcomes = {
                k.split(":", 1)[1]: v for k, v in self._totals.items() if k.startswith("outcome:")
            }
            games = [self._games[g].summary() for g in list(self._order)[-recent_games:]
                     if g in self._games]
            games.reverse()
            games_with_fb = sum(
                1 for g in self._order if g in self._games and self._games[g].fallbacks > 0
            )
            return {
                "uptime_s": round(time.time() - self._started_at, 1),
                "games": self._totals["games"],
                "moves": moves,
                # The headline number. Anything above zero means the RL agent
                # stopped driving and the crude heuristic answered instead.
                "fallback_moves": fallback_moves,
                "fallback_rate": round(fallback_moves / moves, 4) if moves else 0.0,
                "games_with_fallback": games_with_fb,
                "over_budget_moves": self._totals["over_budget"],
                "slow_moves": self._totals["slow_moves"],
                "latency_ms": {
                    "p50": _percentile(lat, 50),
                    "p95": _percentile(lat, 95),
                    "p99": _percentile(lat, 99),
                    "max": round(max(lat), 1) if lat else 0.0,
                    "budget": LATENCY_BUDGET_MS,
                },
                "move_sources": sources,
                "outcomes": outcomes,
                "recent_games": games,
            }

    def recent_decisions(self, n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._recent)[-n:][::-1]

    def healthy(self) -> bool:
        with self._lock:
            return self._totals["fallback_moves"] == 0 and self._totals["over_budget"] == 0
