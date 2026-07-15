"""
Combat-oriented action filtering for Blackout inference.

Strategy: **grow then hunt**.
  1. Hard-filter suicidal H2H (equal/longer enemy heads).
  2. Soft-rank remaining moves:
       - Eat food when we are not clearly the longest (need size to attack).
       - When longer than an enemy, prefer contested cells near their head (hunt).
       - Stay away from equal/longer heads (don't get eaten).
       - Prefer open space so hunts don't trap us.
"""

from __future__ import annotations

from collections import deque
from typing import Any, List, Optional, Sequence, Set, Tuple

import numpy as np

# hisss: UP=0, RIGHT=1, DOWN=2, LEFT=3
_ACTION_DELTA = {
    0: (0, 1),
    1: (1, 0),
    2: (0, -1),
    3: (-1, 0),
}


def _xy(pt: Any) -> Tuple[int, int]:
    if isinstance(pt, (tuple, list)):
        return int(pt[0]), int(pt[1])
    return int(pt[0]), int(pt[1])


def _occupied_cells(state: Any, *, ignore_tails: bool = True) -> Set[Tuple[int, int]]:
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


def next_cell(head: Tuple[int, int], action: int) -> Tuple[int, int]:
    dx, dy = _ACTION_DELTA[int(action)]
    return head[0] + dx, head[1] + dy


def is_losing_head_to_head(
    nxt: Tuple[int, int],
    our_len: int,
    enemies: Sequence[Tuple[int, int, int]],
) -> bool:
    """True if ``nxt`` is contested by an equal/longer enemy this turn."""
    for ex, ey, elen in enemies:
        if elen < our_len:
            continue
        if (ex, ey) == nxt:
            return True
        if abs(ex - nxt[0]) + abs(ey - nxt[1]) == 1:
            return True
    return False


def is_winning_head_to_head(
    nxt: Tuple[int, int],
    our_len: int,
    enemies: Sequence[Tuple[int, int, int]],
) -> bool:
    """True if ``nxt`` contests a *shorter* enemy head (we would win the H2H)."""
    for ex, ey, elen in enemies:
        if elen >= our_len:
            continue
        if (ex, ey) == nxt:
            return True
        if abs(ex - nxt[0]) + abs(ey - nxt[1]) == 1:
            return True
    return False


def flood_fill_space(
    start: Tuple[int, int],
    *,
    width: int,
    height: int,
    blocked: Set[Tuple[int, int]],
    limit: int = 120,
) -> int:
    if start in blocked:
        return 0
    sx, sy = start
    if sx < 0 or sy < 0 or sx >= width or sy >= height:
        return 0
    seen = {start}
    q: deque[Tuple[int, int]] = deque([start])
    while q and len(seen) < limit:
        x, y = q.popleft()
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            npt = (nx, ny)
            if npt in seen or npt in blocked:
                continue
            seen.add(npt)
            q.append(npt)
    return len(seen)


def min_distance_to(
    nxt: Tuple[int, int],
    enemies: Sequence[Tuple[int, int, int]],
    *,
    min_len: Optional[int] = None,
    max_len_exclusive: Optional[int] = None,
) -> float:
    """Nearest Manhattan distance to enemies filtered by length bounds."""
    best = float("inf")
    for ex, ey, elen in enemies:
        if min_len is not None and elen < min_len:
            continue
        if max_len_exclusive is not None and elen >= max_len_exclusive:
            continue
        d = abs(ex - nxt[0]) + abs(ey - nxt[1])
        if d < best:
            best = float(d)
    return best


def food_set(state: Any) -> Set[Tuple[int, int]]:
    foods: Set[Tuple[int, int]] = set()
    for fp in state.food_pos or []:
        foods.add((int(fp[0]), int(fp[1])))
    return foods


def filter_survival_actions(
    env: Any,
    pid: int,
    legal: Optional[List[int]] = None,
) -> List[int]:
    """Hard-filter legal actions that lose an immediate head-to-head."""
    if legal is None:
        legal = list(env.available_actions(pid))
    if not legal:
        return []
    state = env.get_state()
    if not state.snakes_alive[pid]:
        return list(legal)
    body = state.snake_pos.get(pid) or []
    if not body:
        return list(legal)
    head = _xy(body[0])
    our_len = int(state.snake_len[pid])
    enemies = _enemy_heads(state, pid)
    safe: List[int] = []
    for a in legal:
        nxt = next_cell(head, a)
        if is_losing_head_to_head(nxt, our_len, enemies):
            continue
        safe.append(int(a))
    return safe if safe else list(legal)


def rank_survival_actions(
    env: Any,
    pid: int,
    *,
    legal: Optional[List[int]] = None,
    q_values: Optional[np.ndarray] = None,
    hunger_health: int = 35,
    strategy: str = "aggressive",
    w_q: float = 1.0,
    w_space: float = 0.04,
    w_danger: float = 0.10,
    w_hunt: float = 0.18,
    w_food: float = 0.20,
) -> List[int]:
    """
    Rank legal actions under survival / grow-then-hunt heuristics.

    ``strategy``:
      - ``aggressive`` (default): eat to get biggest, then contest shorter heads.
      - ``defensive``: avoid food when healthy, keep distance from all threats.
    """
    legal = filter_survival_actions(env, pid, legal)
    if len(legal) <= 1:
        return legal

    strategy = (strategy or "aggressive").strip().lower()
    aggressive = strategy not in {"defensive", "survive", "survival"}

    state = env.get_state()
    body = state.snake_pos.get(pid) or []
    head = _xy(body[0])
    our_len = int(state.snake_len[pid])
    our_health = int(state.snake_health[pid])
    enemies = _enemy_heads(state, pid)
    foods = food_set(state)
    blocked = _occupied_cells(state, ignore_tails=True)
    blocked.discard(head)
    width = int(env.cfg.w)
    height = int(env.cfg.h)

    max_enemy = max((elen for _, _, elen in enemies), default=0)
    # Need food when low health OR not yet longer than every living enemy.
    want_food = our_health < hunger_health or (aggressive and our_len <= max_enemy)
    can_hunt = aggressive and any(elen < our_len for _, _, elen in enemies)

    scored: List[Tuple[float, int]] = []
    for a in legal:
        nxt = next_cell(head, a)
        blocked_after = set(blocked)
        blocked_after.add(nxt)
        space = flood_fill_space(nxt, width=width, height=height, blocked=blocked_after)

        # Stay away from equal/longer heads.
        danger = min_distance_to(nxt, enemies, min_len=our_len)
        if not np.isfinite(danger):
            danger = 8.0
        danger = min(danger, 8.0)

        q = float(q_values[a]) if q_values is not None else 0.0

        food_term = 0.0
        if nxt in foods:
            food_term = w_food if want_food else (-0.5 * w_food if aggressive else -w_food)

        hunt_term = 0.0
        if can_hunt:
            # Reward closing on / contesting shorter prey.
            if is_winning_head_to_head(nxt, our_len, enemies):
                hunt_term = w_hunt * 1.5
            else:
                prey_dist = min_distance_to(nxt, enemies, max_len_exclusive=our_len)
                if np.isfinite(prey_dist):
                    # Closer to prey ⇒ higher score (cap at distance 8).
                    hunt_term = w_hunt * (8.0 - min(prey_dist, 8.0)) / 8.0

        score = (
            w_q * q
            + w_space * float(np.log1p(space))
            + w_danger * danger
            + food_term
            + hunt_term
        )
        scored.append((score, int(a)))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [a for _, a in scored]


def select_survival_action(
    env: Any,
    pid: int,
    q_values: np.ndarray,
    *,
    legal: Optional[List[int]] = None,
    hunger_health: int = 35,
    strategy: str = "aggressive",
) -> int:
    """Pick the top-ranked combat/survival action (Q-aware)."""
    ranked = rank_survival_actions(
        env,
        pid,
        legal=legal,
        q_values=q_values,
        hunger_health=hunger_health,
        strategy=strategy,
    )
    if not ranked:
        legal = list(env.available_actions(pid))
        return int(legal[0]) if legal else 0
    return int(ranked[0])
