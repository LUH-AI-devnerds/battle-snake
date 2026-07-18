"""
Pure-Python safe move selection from Battlesnake JSON (no hisss required).

Used as a hard filter over the RL policy and as the only fallback path so we
never return a fixed direction like ``up`` into a wall.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

# name -> (dx, dy) in Battlesnake coords (y increases upward)
_DELTAS: Dict[str, Tuple[int, int]] = {
    "up": (0, 1),
    "right": (1, 0),
    "down": (0, -1),
    "left": (-1, 0),
}
_MOVES = ("up", "right", "down", "left")


def _xy(pt: Mapping[str, Any] | Sequence[Any]) -> Tuple[int, int]:
    if isinstance(pt, Mapping):
        return int(pt["x"]), int(pt["y"])
    return int(pt[0]), int(pt[1])


def _body(snake: Mapping[str, Any]) -> List[Tuple[int, int]]:
    body = snake.get("body") or []
    if not body and snake.get("head"):
        body = [snake["head"]]
    out: List[Tuple[int, int]] = []
    for seg in body:
        x, y = _xy(seg)
        if x < 0 or y < 0:
            continue
        if not out or out[-1] != (x, y):
            out.append((x, y))
    return out


def _board_size(payload: Mapping[str, Any]) -> Tuple[int, int]:
    board = payload.get("board") or payload
    return int(board["width"]), int(board["height"])


def occupied_cells(
    payload: Mapping[str, Any],
    *,
    ignore_tails: bool = True,
) -> Set[Tuple[int, int]]:
    board = payload.get("board") or payload
    snakes = list(board.get("snakes") or [])
    occ: Set[Tuple[int, int]] = set()
    for snake in snakes:
        health = int(snake.get("health") or 0)
        if health <= 0 or snake.get("elimination") or snake.get("elimination_event"):
            continue
        body = _body(snake)
        if not body:
            continue
        cells = body[:-1] if ignore_tails and len(body) > 1 else body
        occ.update(cells)
    return occ


def legal_moves(payload: Mapping[str, Any]) -> List[str]:
    """Moves that stay on-board, avoid bodies, and do not reverse into the neck."""
    you = payload.get("you") or {}
    body = _body(you)
    if not body:
        return list(_MOVES)
    head = body[0]
    neck = body[1] if len(body) > 1 else None
    width, height = _board_size(payload)
    blocked = occupied_cells(payload, ignore_tails=True)
    blocked.discard(head)

    legal: List[str] = []
    for name, (dx, dy) in _DELTAS.items():
        nx, ny = head[0] + dx, head[1] + dy
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            continue
        if neck is not None and (nx, ny) == neck:
            continue
        if (nx, ny) in blocked:
            continue
        legal.append(name)
    return legal if legal else list(_MOVES)


def flood_fill(
    start: Tuple[int, int],
    *,
    width: int,
    height: int,
    blocked: Set[Tuple[int, int]],
    limit: int = 80,
) -> int:
    if (
        start[0] < 0
        or start[1] < 0
        or start[0] >= width
        or start[1] >= height
        or start in blocked
    ):
        return 0
    seen = {start}
    q: deque[Tuple[int, int]] = deque([start])
    while q and len(seen) < limit:
        x, y = q.popleft()
        for dx, dy in _DELTAS.values():
            nxt = (x + dx, y + dy)
            if (
                nxt[0] < 0
                or nxt[1] < 0
                or nxt[0] >= width
                or nxt[1] >= height
                or nxt in blocked
                or nxt in seen
            ):
                continue
            seen.add(nxt)
            q.append(nxt)
    return len(seen)


def score_move(payload: Mapping[str, Any], move: str) -> float:
    """Prefer open space, then food when hungry / short, then stay off edges."""
    you = payload.get("you") or {}
    body = _body(you)
    if not body:
        return 0.0
    head = body[0]
    dx, dy = _DELTAS[move]
    nxt = (head[0] + dx, head[1] + dy)
    width, height = _board_size(payload)
    blocked = occupied_cells(payload, ignore_tails=True)
    blocked.discard(head)
    blocked_after = set(blocked)
    blocked_after.add(nxt)
    space = flood_fill(nxt, width=width, height=height, blocked=blocked_after)

    board = payload.get("board") or payload
    foods = {_xy(f) for f in (board.get("food") or []) if _xy(f)[0] >= 0}
    health = int(you.get("health") or 100)
    length = int(you.get("length") or len(body))
    want_food = health < 40 or length <= 4
    food_bonus = 3.0 if (nxt in foods and want_food) else (0.5 if nxt in foods else 0.0)

    # Penalize charging a nearby wall in this direction.
    if move == "up":
        wall_dist = height - 1 - head[1]
    elif move == "down":
        wall_dist = head[1]
    elif move == "right":
        wall_dist = width - 1 - head[0]
    else:
        wall_dist = head[0]
    wall_pen = 4.0 if wall_dist <= 1 else (1.5 if wall_dist <= 3 else 0.0)

    # Soft pull toward board center so we do not hug one edge forever.
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    center_bonus = -0.15 * (abs(nxt[0] - cx) + abs(nxt[1] - cy))

    return float(space) + food_bonus - wall_pen + center_bonus


def choose_safe_move(
    payload: Mapping[str, Any],
    *,
    preferred: Optional[str] = None,
    preferred_order: Optional[Sequence[str]] = None,
) -> str:
    """
    Pick a non-suicidal move.

    Rank by space / food / wall distance. ``preferred`` is only a tie-breaker
    among near-equal scores so the model cannot force a wall-charge.
    """
    legal = legal_moves(payload)
    if not legal:
        return "right"

    order = [m for m in (preferred_order or []) if m in legal]
    for m in legal:
        if m not in order:
            order.append(m)

    def key(m: str) -> Tuple[float, int, int]:
        # Higher score wins; preferred gets a small tie bonus; stable by order.
        tie = 1 if preferred is not None and m == preferred else 0
        return (score_move(payload, m), tie, -order.index(m))

    return max(legal, key=key)
