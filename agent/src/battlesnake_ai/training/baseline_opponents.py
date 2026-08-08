"""Faithful ports of the official Blackout baseline agents.

Why this exists: the hand-written bots in ``heuristic_opponents`` read
``env.get_state()``, i.e. the *whole* board. Real opponents see only what fog
of war leaves them (view radius 5). Training and benchmarking against
omniscient opponents measures the wrong thing -- our own benchmark said 58-66%
win rate while the live leaderboard put us below every non-random baseline.

``HungryBaseline`` is a port of ``hungry_agent.py`` from the official starter
(github.com/l-berg/bs-blackout-starter), which ranks well above us on the
leaderboard despite being simple: A* to the nearest remembered food, follow
your own tail if no food is reachable, otherwise any safe direction. It keeps
a memory of food seen earlier and now out of sight, and -- notably -- has no
head-to-head avoidance at all.
"""

from __future__ import annotations

import heapq
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# hisss action convention: UP=0 RIGHT=1 DOWN=2 LEFT=3
_ACTION_DELTA: Dict[int, Tuple[int, int]] = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
_DELTA_ACTION: Dict[Tuple[int, int], int] = {v: k for k, v in _ACTION_DELTA.items()}

DEFAULT_VIEW_RADIUS = 5


def _visible(head: Tuple[int, int], cell: Tuple[int, int], radius: int) -> bool:
    """Starter uses Manhattan distance for the vision mask."""
    return abs(cell[0] - head[0]) + abs(cell[1] - head[1]) <= radius


def _a_star(
    blocked: Set[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    width: int,
    height: int,
) -> Optional[List[Tuple[int, int]]]:
    open_set: List[Tuple[int, Tuple[int, int]]] = [(0, start)]
    g_score: Dict[Tuple[int, int], int] = {start: 0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        cx, cy = current
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            neighbour = (nx, ny)
            if neighbour in blocked and neighbour != goal:
                continue
            new_g = g_score[current] + 1
            if new_g < g_score.get(neighbour, 1 << 30):
                came_from[neighbour] = current
                g_score[neighbour] = new_g
                f = new_g + abs(nx - goal[0]) + abs(ny - goal[1])
                heapq.heappush(open_set, (f, neighbour))
    return None


class HungryBaseline:
    """Port of the starter's HungryAgent, restricted to what fog of war shows.

    One instance per seat: it carries food memory across turns, and the starter
    keys that memory per game, so callers must ``reset()`` between episodes.
    """

    name = "hungry_baseline"

    def __init__(self, view_radius: int = DEFAULT_VIEW_RADIUS) -> None:
        self.view_radius = view_radius
        self._remembered_food: Dict[int, List[Tuple[int, int]]] = {}

    def reset(self) -> None:
        self._remembered_food.clear()

    # ── perception ───────────────────────────────────────────────────────────

    def _obstacles(self, state: Any, head: Tuple[int, int], w: int, h: int) -> Set[Tuple[int, int]]:
        """Snake bodies within view. The starter skips the tail of each snake
        and cannot see body sections outside its vision (they arrive as None)."""
        blocked: Set[Tuple[int, int]] = set()
        for pid, alive in enumerate(state.snakes_alive):
            if not alive:
                continue
            body = list(state.snake_pos.get(pid) or [])
            if not body:
                continue
            for seg in body[:-1]:  # tail moves out of the way
                cell = (int(seg[0]), int(seg[1]))
                if _visible(head, cell, self.view_radius):
                    blocked.add(cell)
        return blocked

    def _update_food_memory(self, pid: int, state: Any, head: Tuple[int, int]) -> List[Tuple[int, int]]:
        remembered = self._remembered_food.setdefault(pid, [])
        # Drop remembered food that is now in view but gone (eaten by someone).
        kept = [f for f in remembered if not _visible(head, f, self.view_radius)]
        for fp in (state.food_pos or []):
            cell = (int(fp[0]), int(fp[1]))
            if _visible(head, cell, self.view_radius) and cell not in kept:
                kept.append(cell)
        self._remembered_food[pid] = kept
        return kept

    # ── policy ───────────────────────────────────────────────────────────────

    def select_action(self, env: Any, pid: int) -> int:
        legal = list(env.available_actions(pid))
        state = env.get_state()
        body = list(state.snake_pos.get(pid) or [])
        if not body:
            return int(random.choice(legal)) if legal else 0

        head = (int(body[0][0]), int(body[0][1]))
        w, h = int(env.cfg.w), int(env.cfg.h)
        blocked = self._obstacles(state, head, w, h)
        blocked.discard(head)
        foods = self._update_food_memory(pid, state, head)

        # A* to the closest reachable remembered food.
        best_dir: Optional[int] = None
        best_len = float("inf")
        for food in foods:
            path = _a_star(blocked, head, food, w, h)
            if path and len(path) > 1 and len(path) < best_len:
                best_len = len(path)
                best_dir = _DELTA_ACTION.get((path[1][0] - head[0], path[1][1] - head[1]))

        # Fallback 1: follow our own tail.
        if best_dir is None and len(body) > 1:
            tail = (int(body[-1][0]), int(body[-1][1]))
            path = _a_star(blocked, head, tail, w, h)
            if path and len(path) > 1:
                best_dir = _DELTA_ACTION.get((path[1][0] - head[0], path[1][1] - head[1]))

        # Fallback 2: any direction that is on-board and not a visible obstacle.
        if best_dir is None:
            options = []
            for a, (dx, dy) in _ACTION_DELTA.items():
                nxt = (head[0] + dx, head[1] + dy)
                if 0 <= nxt[0] < w and 0 <= nxt[1] < h and nxt not in blocked:
                    options.append(a)
            best_dir = random.choice(options) if options else 0

        if legal and best_dir not in legal:
            # The starter would walk into it; hisss would reject the joint action,
            # so keep it legal but do not "improve" the choice beyond that.
            return int(random.choice(legal))
        return int(best_dir)


class RandomBaseline:
    """Port of the starter's random_agent: any on-board, non-obstacle move."""

    name = "random_baseline"

    def reset(self) -> None:  # symmetry with HungryBaseline
        return

    def select_action(self, env: Any, pid: int) -> int:
        legal = list(env.available_actions(pid))
        return int(random.choice(legal)) if legal else 0


ALL_BASELINES = {
    "hungry_baseline": HungryBaseline,
    "random_baseline": RandomBaseline,
}


def make_baseline(name: str) -> Any:
    if name not in ALL_BASELINES:
        raise ValueError(f"Unknown baseline: {name!r} (have {sorted(ALL_BASELINES)})")
    return ALL_BASELINES[name]()
