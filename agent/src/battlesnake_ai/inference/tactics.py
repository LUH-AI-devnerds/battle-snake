"""Tactical move selection straight from Battlesnake JSON (no hisss, no model).

This is the JSON-side port of ``training.heuristic_opponents.AggressiveHunterBot``
with the head-to-head guard of ``CautiousBot``.  In 4-player restricted_standard
evaluation that bot wins ~75% of games while the trained nets win 5-20%, so it
runs as the primary decision maker at inference and the policy is only used to
break near-ties.

Everything here works on *visible* information only, which is what fog-of-war
leaves in the ``/move`` payload.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

_DELTAS: Dict[str, Tuple[int, int]] = {
    "up": (0, 1),
    "right": (1, 0),
    "down": (0, -1),
    "left": (-1, 0),
}
_MOVES = ("up", "right", "down", "left")


def _xy(pt: Any) -> Optional[Tuple[int, int]]:
    """Coordinates from a point, or None if it is missing / malformed.

    Returning None rather than raising matters: this runs on the live request
    path, and an exception here drops the whole move to the crude last-resort
    heuristic. Fogged and eliminated snakes carry nulls in these fields.
    """
    try:
        if isinstance(pt, Mapping):
            x, y = pt.get("x"), pt.get("y")
        elif isinstance(pt, Sequence) and not isinstance(pt, (str, bytes)) and len(pt) >= 2:
            x, y = pt[0], pt[1]
        else:
            return None
        if x is None or y is None:
            return None
        return int(x), int(y)
    except (TypeError, ValueError):
        return None


def _body(snake: Mapping[str, Any]) -> List[Tuple[int, int]]:
    body = snake.get("body") or []
    if not body and snake.get("head"):
        body = [snake["head"]]
    out: List[Tuple[int, int]] = []
    for seg in body:
        pt = _xy(seg)
        if pt is None:
            continue
        x, y = pt
        if x < 0 or y < 0:  # fog-of-war placeholder
            continue
        if not out or out[-1] != (x, y):
            out.append((x, y))
    return out


def _food_cells(payload: Mapping[str, Any]) -> Set[Tuple[int, int]]:
    """On-board food cells, skipping nulls and fog-of-war (-1) placeholders."""
    out: Set[Tuple[int, int]] = set()
    for f in (_board(payload).get("food") or []):
        pt = _xy(f)
        if pt is not None and pt[0] >= 0 and pt[1] >= 0:
            out.add(pt)
    return out


def _alive(snake: Mapping[str, Any]) -> bool:
    if snake.get("elimination") or snake.get("elimination_event"):
        return False
    return int(snake.get("health") or 0) > 0


def _board(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return payload.get("board") or payload


def _dims(payload: Mapping[str, Any]) -> Tuple[int, int]:
    """Board size, defaulting to the Blackout 15x15 board if absent/malformed.

    A missing dimension used to raise straight out of the decision path.
    Guessing the standard board is strictly better than losing the move.
    """
    b = _board(payload)
    try:
        w = int(b.get("width") or 15)
        h = int(b.get("height") or 15)
    except (TypeError, ValueError):
        return 15, 15
    return (w if w > 0 else 15), (h if h > 0 else 15)


def occupied(payload: Mapping[str, Any], *, ignore_tails: bool = True) -> Set[Tuple[int, int]]:
    occ: Set[Tuple[int, int]] = set()
    for snake in _board(payload).get("snakes") or []:
        if not _alive(snake):
            continue
        body = _body(snake)
        if not body:
            continue
        occ.update(body[:-1] if ignore_tails and len(body) > 1 else body)
    return occ


def enemy_heads(payload: Mapping[str, Any]) -> List[Tuple[int, int, int]]:
    """(x, y, length) for every visible living opponent."""
    you_id = str((payload.get("you") or {}).get("id", ""))
    out: List[Tuple[int, int, int]] = []
    for snake in _board(payload).get("snakes") or []:
        if str(snake.get("id", "")) == you_id or not _alive(snake):
            continue
        body = _body(snake)
        if not body:
            continue
        out.append((body[0][0], body[0][1], int(snake.get("length") or len(body))))
    return out


def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def flood_fill(
    start: Tuple[int, int],
    *,
    width: int,
    height: int,
    blocked: Set[Tuple[int, int]],
    limit: int = 200,
) -> int:
    if start in blocked or not _in_bounds(start[0], start[1], width, height):
        return 0
    seen = {start}
    q: deque[Tuple[int, int]] = deque([start])
    while q and len(seen) < limit:
        x, y = q.popleft()
        for dx, dy in _DELTAS.values():
            nxt = (x + dx, y + dy)
            if not _in_bounds(nxt[0], nxt[1], width, height) or nxt in seen or nxt in blocked:
                continue
            seen.add(nxt)
            q.append(nxt)
    return len(seen)


def bfs_distance(
    start: Tuple[int, int],
    targets: Set[Tuple[int, int]],
    *,
    width: int,
    height: int,
    blocked: Set[Tuple[int, int]],
    limit: int = 200,
) -> Optional[int]:
    if start in targets:
        return 0
    if start in blocked or not _in_bounds(start[0], start[1], width, height):
        return None
    seen = {start}
    q: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
    while q and len(seen) < limit:
        (x, y), dist = q.popleft()
        for dx, dy in _DELTAS.values():
            nxt = (x + dx, y + dy)
            if not _in_bounds(nxt[0], nxt[1], width, height) or nxt in seen or nxt in blocked:
                continue
            if nxt in targets:
                return dist + 1
            seen.add(nxt)
            q.append((nxt, dist + 1))
    return None


def legal_moves(payload: Mapping[str, Any]) -> List[str]:
    """On-board moves that avoid bodies and do not reverse into the neck."""
    you = payload.get("you") or {}
    body = _body(you)
    if not body:
        return list(_MOVES)
    head = body[0]
    neck = body[1] if len(body) > 1 else None
    width, height = _dims(payload)
    blocked = occupied(payload, ignore_tails=True)
    blocked.discard(head)

    out: List[str] = []
    for name, (dx, dy) in _DELTAS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if not _in_bounds(nxt[0], nxt[1], width, height):
            continue
        if neck is not None and nxt == neck:
            continue
        if nxt in blocked:
            continue
        out.append(name)
    return out


def safe_moves(payload: Mapping[str, Any]) -> List[str]:
    """Legal moves that also avoid a head-to-head with an equal/longer snake."""
    legal = legal_moves(payload)
    if not legal:
        return []
    you = payload.get("you") or {}
    body = _body(you)
    if not body:
        return legal
    head = body[0]
    our_len = int(you.get("length") or len(body))
    enemies = enemy_heads(payload)

    safe: List[str] = []
    for name in legal:
        dx, dy = _DELTAS[name]
        nx, ny = head[0] + dx, head[1] + dy
        risky = any(
            elen >= our_len and abs(ex - nx) + abs(ey - ny) <= 1 for ex, ey, elen in enemies
        )
        if not risky:
            safe.append(name)
    return safe or legal


def score_move(payload: Mapping[str, Any], move: str) -> float:
    """AggressiveHunter scoring: space, food urgency, threat distance, hunting."""
    you = payload.get("you") or {}
    body = _body(you)
    if not body:
        return 0.0
    head = body[0]
    our_len = int(you.get("length") or len(body))
    health = int(you.get("health") or 100)
    width, height = _dims(payload)

    blocked = occupied(payload, ignore_tails=True)
    blocked.discard(head)

    dx, dy = _DELTAS[move]
    nxt = (head[0] + dx, head[1] + dy)
    blocked_after = set(blocked)
    blocked_after.add(nxt)

    enemies = enemy_heads(payload)
    foods = _food_cells(payload)
    max_enemy_len = max((elen for _, _, elen in enemies), default=0)
    can_hunt = any(elen < our_len for _, _, elen in enemies)
    want_food = health < 60 or our_len <= max_enemy_len

    score = 0.0

    # 1. Reachable space — the dominant survival term.
    space = flood_fill(nxt, width=width, height=height, blocked=blocked_after)
    score += math.log1p(space) * 0.5
    # Hard penalty for moving into a pocket we cannot fit in.
    if space <= our_len:
        score -= 3.0

    # 2. Food: eat to grow, urgently when hungry or out-lengthed.
    if nxt in foods:
        score += 5.0 if want_food else 1.0
    elif want_food and foods:
        blocked_bfs = set(blocked)
        blocked_bfs.add(head)
        dist = bfs_distance(nxt, foods, width=width, height=height, blocked=blocked_bfs)
        if dist is not None:
            score += max(0, 12 - dist) * 0.4

    # 3. Keep away from equal/longer heads; close on shorter ones once we lead.
    for ex, ey, elen in enemies:
        dist = abs(ex - nxt[0]) + abs(ey - nxt[1])
        if elen >= our_len:
            score += min(dist, 8) * 0.3
        elif can_hunt and our_len > max_enemy_len:
            score -= min(dist, 8) * 0.3
            if dist <= 1:
                score += 4.0

    return float(score)


def choose_move(
    payload: Mapping[str, Any],
    *,
    preferred: Optional[str] = None,
    tie_eps: float = 0.01,
) -> Tuple[str, Dict[str, Any]]:
    """Pick a move; ``preferred`` (the policy's choice) only breaks near-ties.

    Returns ``(move, debug)``.
    """
    candidates = safe_moves(payload)
    if not candidates:
        candidates = legal_moves(payload) or list(_MOVES)

    scores = {m: score_move(payload, m) for m in candidates}
    best = max(scores.values())
    top = [m for m, s in scores.items() if s >= best - tie_eps]
    move = top[0]
    used_model = False
    if preferred in top and len(top) > 1:
        move = preferred
        used_model = True
    return move, {"scores": {m: round(s, 3) for m, s in scores.items()},
                  "candidates": candidates,
                  "tie_break_model": used_model}
