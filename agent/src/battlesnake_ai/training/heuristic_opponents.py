"""Heuristic opponent bots for training.

Each bot operates on the raw ``hisss`` game state and selects actions
using hand-crafted strategies.  They provide structured, deterministic
challenges at varying difficulty levels — much stronger training signal
than uniform-random opponents.

Reuses utilities from :mod:`battlesnake_ai.inference.survival`.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ── Movement helpers ──────────────────────────────────────────────────────────

# hisss action convention:  UP=0  RIGHT=1  DOWN=2  LEFT=3
_ACTION_DELTA: Dict[int, Tuple[int, int]] = {
    0: (0, 1),
    1: (1, 0),
    2: (0, -1),
    3: (-1, 0),
}

_ALL_ACTIONS = [0, 1, 2, 3]


def _xy(pt: Any) -> Tuple[int, int]:
    """Extract (x, y) from a hisss coordinate."""
    if isinstance(pt, (tuple, list)):
        return int(pt[0]), int(pt[1])
    return int(pt[0]), int(pt[1])


def _next_cell(head: Tuple[int, int], action: int) -> Tuple[int, int]:
    dx, dy = _ACTION_DELTA[action]
    return head[0] + dx, head[1] + dy


def _occupied_cells(state: Any, *, ignore_tails: bool = True) -> Set[Tuple[int, int]]:
    """Return the set of cells occupied by any alive snake body."""
    occ: Set[Tuple[int, int]] = set()
    for pid, alive in enumerate(state.snakes_alive):
        if not alive:
            continue
        body = state.snake_pos.get(pid) or []
        if not body:
            continue
        cells = body[:-1] if ignore_tails and len(body) > 1 else body
        for seg in cells:
            occ.add(_xy(seg))
    return occ


def _enemy_heads(state: Any, our_pid: int) -> List[Tuple[int, int, int]]:
    """Return list of (hx, hy, length) for all alive enemies."""
    out: List[Tuple[int, int, int]] = []
    for pid, alive in enumerate(state.snakes_alive):
        if not alive or pid == our_pid:
            continue
        body = state.snake_pos.get(pid) or []
        if not body:
            continue
        hx, hy = _xy(body[0])
        out.append((hx, hy, int(state.snake_len[pid])))
    return out


def _food_positions(state: Any) -> Set[Tuple[int, int]]:
    foods: Set[Tuple[int, int]] = set()
    for fp in state.food_pos or []:
        foods.add((int(fp[0]), int(fp[1])))
    return foods


def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def _flood_fill(
    start: Tuple[int, int],
    *,
    width: int,
    height: int,
    blocked: Set[Tuple[int, int]],
    limit: int = 200,
) -> int:
    """Count reachable cells from *start* via BFS, up to *limit*."""
    if start in blocked:
        return 0
    sx, sy = start
    if not _in_bounds(sx, sy, width, height):
        return 0
    seen = {start}
    q: deque[Tuple[int, int]] = deque([start])
    while q and len(seen) < limit:
        x, y = q.popleft()
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny, width, height):
                continue
            npt = (nx, ny)
            if npt in seen or npt in blocked:
                continue
            seen.add(npt)
            q.append(npt)
    return len(seen)


def _bfs_distance(
    start: Tuple[int, int],
    targets: Set[Tuple[int, int]],
    *,
    width: int,
    height: int,
    blocked: Set[Tuple[int, int]],
    limit: int = 200,
) -> Optional[Tuple[int, int, int]]:
    """BFS shortest path from *start* to any cell in *targets*.

    Returns ``(target_x, target_y, distance)`` or ``None``.
    """
    if start in targets:
        return (start[0], start[1], 0)
    if start in blocked:
        return None
    sx, sy = start
    if not _in_bounds(sx, sy, width, height):
        return None
    seen = {start}
    q: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
    while q and len(seen) < limit:
        (x, y), dist = q.popleft()
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny, width, height):
                continue
            npt = (nx, ny)
            if npt in seen or npt in blocked:
                continue
            if npt in targets:
                return (nx, ny, dist + 1)
            seen.add(npt)
            q.append((npt, dist + 1))
    return None


def _safe_actions(
    env: Any,
    pid: int,
    state: Any,
    head: Tuple[int, int],
    our_len: int,
    blocked: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> List[int]:
    """Return legal actions that don't walk into walls, bodies, or losing H2H."""
    legal = list(env.available_actions(pid))
    enemies = _enemy_heads(state, pid)
    safe: List[int] = []
    for a in legal:
        nxt = _next_cell(head, a)
        nx, ny = nxt
        # Out of bounds
        if not _in_bounds(nx, ny, width, height):
            continue
        # Walk into body
        if nxt in blocked:
            continue
        # Losing head-to-head: adjacent to equal/longer enemy head
        losing_h2h = False
        for ex, ey, elen in enemies:
            if elen < our_len:
                continue
            # Enemy could move to our target cell
            if abs(ex - nx) + abs(ey - ny) <= 1:
                losing_h2h = True
                break
        if not losing_h2h:
            safe.append(a)
    return safe if safe else legal


# ── Base class ────────────────────────────────────────────────────────────────


class HeuristicBot(ABC):
    """Interface for rule-based snake opponents."""

    name: str = "base"

    @abstractmethod
    def select_action(self, env: Any, pid: int) -> int:
        """Choose an action for snake *pid* given the current environment state."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── Bot implementations ──────────────────────────────────────────────────────


class RandomBot(HeuristicBot):
    """Uniform random from legal actions.  Easiest baseline."""

    name = "random"

    def select_action(self, env: Any, pid: int) -> int:
        legal = list(env.available_actions(pid))
        return int(random.choice(legal)) if legal else 0


class FloodFillBot(HeuristicBot):
    """Pick the action that maximises reachable space, with food tie-breaking.

    Avoids cornering itself — a strong defensive strategy — but also
    moves toward food when two moves offer similar space.
    """

    name = "flood_fill"

    def select_action(self, env: Any, pid: int) -> int:
        state = env.get_state()
        body = state.snake_pos.get(pid) or []
        if not body:
            return RandomBot().select_action(env, pid)
        head = _xy(body[0])
        our_len = int(state.snake_len[pid])
        our_health = int(state.snake_health[pid])
        width, height = int(env.cfg.w), int(env.cfg.h)
        blocked = _occupied_cells(state, ignore_tails=True)
        blocked.discard(head)

        safe = _safe_actions(env, pid, state, head, our_len, blocked, width, height)
        if not safe:
            legal = list(env.available_actions(pid))
            return int(random.choice(legal)) if legal else 0

        foods = _food_positions(state)

        # Score each action: primary=space, secondary=food proximity
        scored: List[Tuple[float, int]] = []
        for a in safe:
            nxt = _next_cell(head, a)
            blocked_after = set(blocked)
            blocked_after.add(nxt)
            space = _flood_fill(nxt, width=width, height=height, blocked=blocked_after)
            score = float(space)

            # Bonus for stepping onto food
            if nxt in foods:
                score += 3.0
            elif foods and our_health < 50:
                # BFS to nearest food — small bonus for being closer
                blocked_bfs = set(blocked)
                blocked_bfs.add(head)
                result = _bfs_distance(nxt, foods, width=width, height=height, blocked=blocked_bfs)
                if result is not None:
                    _, _, dist = result
                    score += max(0, 10 - dist) * 0.3

            scored.append((score, a))

        scored.sort(key=lambda t: t[0], reverse=True)
        best_score = scored[0][0]
        top = [a for s, a in scored if s >= best_score - 0.5]
        return int(random.choice(top))


class FoodChaserBot(HeuristicBot):
    """BFS to nearest food; eats aggressively to grow.

    Falls back to flood-fill when no food is reachable.
    """

    name = "food_chaser"

    def select_action(self, env: Any, pid: int) -> int:
        state = env.get_state()
        body = state.snake_pos.get(pid) or []
        if not body:
            return RandomBot().select_action(env, pid)
        head = _xy(body[0])
        our_len = int(state.snake_len[pid])
        width, height = int(env.cfg.w), int(env.cfg.h)
        blocked = _occupied_cells(state, ignore_tails=True)
        blocked.discard(head)

        safe = _safe_actions(env, pid, state, head, our_len, blocked, width, height)
        if not safe:
            legal = list(env.available_actions(pid))
            return int(random.choice(legal)) if legal else 0

        foods = _food_positions(state)
        if foods:
            # For each safe action, compute BFS distance to nearest food
            best_dist = float("inf")
            best_actions: List[int] = []
            for a in safe:
                nxt = _next_cell(head, a)
                if nxt in foods:
                    # Immediate food — best possible
                    if 0 < best_dist:
                        best_dist = 0
                        best_actions = [a]
                    else:
                        best_actions.append(a)
                    continue
                blocked_after = set(blocked)
                blocked_after.add(head)  # Our old head is now body
                result = _bfs_distance(
                    nxt, foods, width=width, height=height, blocked=blocked_after
                )
                if result is not None:
                    _, _, dist = result
                    if dist < best_dist:
                        best_dist = dist
                        best_actions = [a]
                    elif dist == best_dist:
                        best_actions.append(a)
            if best_actions:
                return int(random.choice(best_actions))

        # Fallback: flood-fill
        return FloodFillBot().select_action(env, pid)


class AggressiveHunterBot(HeuristicBot):
    """Eat-first, hunt-second strategy.

    Strategy:
      1. Eat food aggressively to grow (like FoodChaser).
      2. Once strictly longer than an enemy, chase for H2H kills.
      3. Always avoid losing head-to-heads with longer/equal snakes.
      4. Maintain space awareness to avoid getting trapped.
    """

    name = "aggressive_hunter"

    def select_action(self, env: Any, pid: int) -> int:
        state = env.get_state()
        body = state.snake_pos.get(pid) or []
        if not body:
            return RandomBot().select_action(env, pid)
        head = _xy(body[0])
        our_len = int(state.snake_len[pid])
        our_health = int(state.snake_health[pid])
        width, height = int(env.cfg.w), int(env.cfg.h)
        blocked = _occupied_cells(state, ignore_tails=True)
        blocked.discard(head)

        enemies = _enemy_heads(state, pid)
        foods = _food_positions(state)
        safe = _safe_actions(env, pid, state, head, our_len, blocked, width, height)
        if not safe:
            legal = list(env.available_actions(pid))
            return int(random.choice(legal)) if legal else 0

        max_enemy_len = max((elen for _, _, elen in enemies), default=0)
        # We only hunt when strictly longer than at least one enemy
        can_hunt = any(elen < our_len for _, _, elen in enemies)
        # We need food when health is low OR we're not the longest
        want_food = our_health < 60 or our_len <= max_enemy_len

        # Score each safe action
        scored: List[Tuple[float, int]] = []
        for a in safe:
            nxt = _next_cell(head, a)
            blocked_after = set(blocked)
            blocked_after.add(nxt)
            score = 0.0

            # 1. Space (survival baseline)
            space = _flood_fill(nxt, width=width, height=height, blocked=blocked_after)
            score += np.log1p(space) * 0.5

            # 2. Food (HIGH priority — eat to grow first)
            if nxt in foods:
                if want_food:
                    score += 5.0  # Very high priority
                else:
                    score += 1.0  # Still decent when not hungry
            elif want_food and foods:
                # BFS to nearest food
                blocked_bfs = set(blocked)
                blocked_bfs.add(head)
                result = _bfs_distance(nxt, foods, width=width, height=height, blocked=blocked_bfs)
                if result is not None:
                    _, _, dist = result
                    score += max(0, 12 - dist) * 0.4

            # 3. Distance from threats (equal/longer enemies)
            for ex, ey, elen in enemies:
                dist = abs(ex - nxt[0]) + abs(ey - nxt[1])
                if elen >= our_len:
                    # Run away from threats
                    score += min(dist, 8) * 0.3
                elif can_hunt and our_len > max_enemy_len:
                    # Only actively hunt once we're the longest
                    score -= min(dist, 8) * 0.3
                    # Immediate H2H win bonus
                    if dist <= 1:
                        score += 4.0

            scored.append((score, a))

        scored.sort(key=lambda t: t[0], reverse=True)
        best_score = scored[0][0]
        top = [a for s, a in scored if s >= best_score - 0.01]
        return int(random.choice(top))


class CautiousBot(HeuristicBot):
    """Balanced strategy: flood-fill + food when hungry + strict H2H avoidance.

    The hardest heuristic bot — plays conservatively, eats to grow,
    never takes risky head-to-heads, and maximizes board control.
    """

    name = "cautious"

    def select_action(self, env: Any, pid: int) -> int:
        state = env.get_state()
        body = state.snake_pos.get(pid) or []
        if not body:
            return RandomBot().select_action(env, pid)
        head = _xy(body[0])
        our_len = int(state.snake_len[pid])
        our_health = int(state.snake_health[pid])
        width, height = int(env.cfg.w), int(env.cfg.h)
        blocked = _occupied_cells(state, ignore_tails=True)
        blocked.discard(head)

        enemies = _enemy_heads(state, pid)
        foods = _food_positions(state)
        safe = _safe_actions(env, pid, state, head, our_len, blocked, width, height)
        if not safe:
            legal = list(env.available_actions(pid))
            return int(random.choice(legal)) if legal else 0

        # Score each safe action
        scored: List[Tuple[float, int]] = []
        for a in safe:
            nxt = _next_cell(head, a)
            blocked_after = set(blocked)
            blocked_after.add(nxt)
            score = 0.0

            # 1. Space (primary objective)
            space = _flood_fill(nxt, width=width, height=height, blocked=blocked_after)
            score += np.log1p(space) * 1.0

            # 2. Stay away from ALL enemy heads (conservative)
            min_enemy_dist = float("inf")
            for ex, ey, elen in enemies:
                dist = abs(ex - nxt[0]) + abs(ey - nxt[1])
                if dist < min_enemy_dist:
                    min_enemy_dist = dist
                # Extra penalty for being close to longer enemies
                if elen >= our_len and dist <= 2:
                    score -= 2.0
            if np.isfinite(min_enemy_dist):
                score += min(min_enemy_dist, 6) * 0.2

            # 3. Food when health is critical or when we're not the longest
            max_enemy_len = max((elen for _, _, elen in enemies), default=0)
            want_food = our_health < 40 or our_len <= max_enemy_len
            if nxt in foods:
                if want_food:
                    score += 2.5
                    # Extra urgency when starving
                    if our_health < 20:
                        score += 2.0
                else:
                    score += 0.1  # Tiny bonus — don't avoid food on purpose

            # 4. Center preference (avoid edges when possible)
            cx, cy = width / 2.0, height / 2.0
            edge_dist = min(nxt[0], nxt[1], width - 1 - nxt[0], height - 1 - nxt[1])
            score += edge_dist * 0.1

            # 5. Opportunistic kill (only if safe — we're strictly longer)
            for ex, ey, elen in enemies:
                if elen < our_len:
                    dist = abs(ex - nxt[0]) + abs(ey - nxt[1])
                    if dist <= 1:
                        score += 1.5  # Take safe kills

            scored.append((score, a))

        scored.sort(key=lambda t: t[0], reverse=True)
        best_score = scored[0][0]
        top = [a for s, a in scored if s >= best_score - 0.01]
        return int(random.choice(top))


# ── Utility ───────────────────────────────────────────────────────────────────

ALL_HEURISTIC_BOTS: List[HeuristicBot] = [
    RandomBot(),
    FloodFillBot(),
    FoodChaserBot(),
    AggressiveHunterBot(),
    CautiousBot(),
]


def get_bot_by_name(name: str) -> HeuristicBot:
    """Look up a heuristic bot by its ``name`` attribute."""
    for bot in ALL_HEURISTIC_BOTS:
        if bot.name == name:
            return bot
    raise ValueError(f"Unknown heuristic bot: {name!r}")
