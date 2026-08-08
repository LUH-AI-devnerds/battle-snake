"""Lookahead search over the Blackout board.

Rationale: the served policy answers in ~15 ms against a 500 ms budget, so 97%
of the available compute was going unused while stronger opponents spend theirs
on search. This module simulates the game forward and picks the move that
survives the opponents' best replies, using Voronoi board control as the
evaluation -- the metric that decides most Battlesnake games.

Everything works on the plain ``/move`` JSON, so it sees exactly what fog of
war leaves us and needs no simulator.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from battlesnake_ai.inference.tactics import (
    _body,
    _dims,
    _food_cells,
    _alive,
    _board,
)

_DELTAS: Dict[str, Tuple[int, int]] = {
    "up": (0, 1), "right": (1, 0), "down": (0, -1), "left": (-1, 0),
}
_MOVES = ("up", "right", "down", "left")

# Evaluation weights. Voronoi control dominates: owning more of the board than
# everyone else is what converts into survival and food access later.
W_CONTROL = 1.0
W_LENGTH = 6.0
W_FOOD = 1.2
W_TRAPPED = 120.0
W_DEAD = 10000.0


class SimSnake:
    __slots__ = ("sid", "body", "health", "length", "alive")

    def __init__(self, sid: str, body: List[Tuple[int, int]], health: int, length: int) -> None:
        self.sid = sid
        self.body = body
        self.health = health
        self.length = length
        self.alive = True

    def copy(self) -> "SimSnake":
        s = SimSnake(self.sid, list(self.body), self.health, self.length)
        s.alive = self.alive
        return s

    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]


class SimBoard:
    """Minimal Battlesnake forward model: simultaneous moves, then collisions."""

    __slots__ = ("w", "h", "snakes", "food", "me")

    def __init__(self, w: int, h: int, snakes: List[SimSnake], food: Set[Tuple[int, int]], me: str) -> None:
        self.w = w
        self.h = h
        self.snakes = snakes
        self.food = food
        self.me = me

    def copy(self) -> "SimBoard":
        return SimBoard(self.w, self.h, [s.copy() for s in self.snakes], set(self.food), self.me)

    def get(self, sid: str) -> Optional[SimSnake]:
        for s in self.snakes:
            if s.sid == sid:
                return s
        return None

    def occupied(self, *, ignore_tails: bool = True) -> Set[Tuple[int, int]]:
        cells: Set[Tuple[int, int]] = set()
        for s in self.snakes:
            if not s.alive or not s.body:
                continue
            cells.update(s.body[:-1] if ignore_tails and len(s.body) > 1 else s.body)
        return cells

    def legal_moves(self, sid: str) -> List[str]:
        s = self.get(sid)
        if s is None or not s.alive or not s.body:
            return []
        head = s.head
        neck = s.body[1] if len(s.body) > 1 else None
        blocked = self.occupied(ignore_tails=True)
        blocked.discard(head)
        out: List[str] = []
        for name, (dx, dy) in _DELTAS.items():
            nxt = (head[0] + dx, head[1] + dy)
            if not (0 <= nxt[0] < self.w and 0 <= nxt[1] < self.h):
                continue
            if neck is not None and nxt == neck:
                continue
            if nxt in blocked:
                continue
            out.append(name)
        return out

    def step(self, moves: Mapping[str, str]) -> "SimBoard":
        """Apply one simultaneous turn. Mirrors standard Battlesnake rules."""
        nb = self.copy()
        heads: Dict[str, Tuple[int, int]] = {}

        for s in nb.snakes:
            if not s.alive:
                continue
            mv = moves.get(s.sid)
            if mv is None:
                # No decision for this snake (out of view): keep it still-ish by
                # repeating its last direction, which is the best guess we have.
                if len(s.body) > 1:
                    dx = s.body[0][0] - s.body[1][0]
                    dy = s.body[0][1] - s.body[1][1]
                else:
                    dx, dy = 0, 1
            else:
                dx, dy = _DELTAS[mv]
            nxt = (s.head[0] + dx, s.head[1] + dy)
            heads[s.sid] = nxt
            s.body.insert(0, nxt)
            s.health -= 1

        # Food: grow (keep tail) if the new head landed on food.
        eaten: Set[Tuple[int, int]] = set()
        for s in nb.snakes:
            if not s.alive:
                continue
            if s.head in nb.food:
                s.health = 100
                s.length += 1
                eaten.add(s.head)
            else:
                if len(s.body) > 1:
                    s.body.pop()
        nb.food -= eaten

        # Deaths: wall, starvation, body collision, then head-to-head by length.
        for s in nb.snakes:
            if not s.alive:
                continue
            hx, hy = s.head
            if not (0 <= hx < nb.w and 0 <= hy < nb.h) or s.health <= 0:
                s.alive = False

        bodies: Dict[Tuple[int, int], int] = {}
        for s in nb.snakes:
            if not s.alive:
                continue
            for seg in s.body[1:]:
                bodies[seg] = bodies.get(seg, 0) + 1

        for s in nb.snakes:
            if not s.alive:
                continue
            if s.head in bodies:
                s.alive = False

        for s in nb.snakes:
            if not s.alive:
                continue
            for o in nb.snakes:
                if o is s or not o.alive or o.head != s.head:
                    continue
                if o.length >= s.length:
                    s.alive = False
                    break
        return nb


def board_from_payload(payload: Mapping[str, Any]) -> Optional[SimBoard]:
    you = payload.get("you") or {}
    you_id = str(you.get("id", ""))
    if not you_id:
        return None
    w, h = _dims(payload)
    snakes: List[SimSnake] = []
    seen: Set[str] = set()
    for raw in (_board(payload).get("snakes") or []):
        if not isinstance(raw, Mapping) or not _alive(raw):
            continue
        sid = str(raw.get("id", ""))
        if not sid or sid in seen:
            continue
        body = _body(raw)
        if not body:
            continue
        seen.add(sid)
        snakes.append(SimSnake(sid, body, int(raw.get("health") or 100),
                               int(raw.get("length") or len(body))))
    if you_id not in seen:
        body = _body(you)
        if not body:
            return None
        snakes.append(SimSnake(you_id, body, int(you.get("health") or 100),
                               int(you.get("length") or len(body))))
    return SimBoard(w, h, snakes, set(_food_cells(payload)), you_id)


def voronoi(board: SimBoard) -> Dict[str, int]:
    """Cells each snake reaches strictly before every other snake.

    Board control, not raw reachable space: a cell an opponent gets to first is
    not really ours. This is the single most predictive cheap feature in
    Battlesnake, and it is what separates holding a region from merely
    standing in it.
    """
    blocked = board.occupied(ignore_tails=True)
    best: Dict[Tuple[int, int], Tuple[int, Optional[str]]] = {}
    q: deque = deque()
    for s in board.snakes:
        if not s.alive or not s.body:
            continue
        q.append((s.head, s.sid, 0))
        best[s.head] = (0, s.sid)

    while q:
        (x, y), sid, dist = q.popleft()
        cur = best.get((x, y))
        if cur is not None and (cur[0] < dist or (cur[0] == dist and cur[1] != sid)):
            continue
        for dx, dy in _DELTAS.values():
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < board.w and 0 <= nxt[1] < board.h) or nxt in blocked:
                continue
            nd = dist + 1
            prev = best.get(nxt)
            if prev is None:
                best[nxt] = (nd, sid)
                q.append((nxt, sid, nd))
            elif nd < prev[0]:
                best[nxt] = (nd, sid)
                q.append((nxt, sid, nd))
            elif nd == prev[0] and prev[1] != sid:
                # Contested: neither side owns it.
                best[nxt] = (nd, None)

    counts: Dict[str, int] = {s.sid: 0 for s in board.snakes}
    for _, (_, owner) in best.items():
        if owner is not None and owner in counts:
            counts[owner] += 1
    return counts


def _nearest_food_dist(board: SimBoard, s: SimSnake) -> Optional[int]:
    if not board.food:
        return None
    blocked = board.occupied(ignore_tails=True)
    blocked.discard(s.head)
    seen = {s.head}
    q: deque = deque([(s.head, 0)])
    while q:
        (x, y), d = q.popleft()
        for dx, dy in _DELTAS.values():
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < board.w and 0 <= nxt[1] < board.h):
                continue
            if nxt in seen or nxt in blocked:
                continue
            if nxt in board.food:
                return d + 1
            seen.add(nxt)
            q.append((nxt, d + 1))
    return None


def evaluate(board: SimBoard) -> float:
    """Score the position from our perspective. Higher is better."""
    me = board.get(board.me)
    if me is None or not me.alive:
        return -W_DEAD

    others = [s for s in board.snakes if s.alive and s.sid != board.me]
    if not others:
        return W_DEAD

    control = voronoi(board)
    my_area = control.get(board.me, 0)
    best_other_area = max((control.get(o.sid, 0) for o in others), default=0)
    score = W_CONTROL * (my_area - best_other_area)

    longest_other = max((o.length for o in others), default=0)
    score += W_LENGTH * (me.length - longest_other)

    # Space we cannot fit into is a trap, however good it looks otherwise.
    if my_area <= me.length:
        score -= W_TRAPPED

    # Prefer being close to food when we are hungry or out-lengthed.
    if me.health < 55 or me.length <= longest_other:
        d = _nearest_food_dist(board, me)
        if d is None:
            score -= W_FOOD * 12
        else:
            score -= W_FOOD * d
    return score


def _relevant_opponents(board: SimBoard, radius: int = 6) -> List[SimSnake]:
    """Opponents close enough to interact with us within the search horizon."""
    me = board.get(board.me)
    if me is None:
        return []
    out = []
    for s in board.snakes:
        if not s.alive or s.sid == board.me:
            continue
        if abs(s.head[0] - me.head[0]) + abs(s.head[1] - me.head[1]) <= radius:
            out.append(s)
    return out


def _worst_reply(board: SimBoard, my_move: str, depth: int, deadline: float) -> float:
    """Value of playing ``my_move``, assuming nearby opponents reply worst-case."""
    opponents = _relevant_opponents(board)
    if not opponents:
        nb = board.step({board.me: my_move})
        return _search(nb, depth - 1, deadline)

    # Paranoid over the closest opponent only; enumerating every opponent's
    # moves explodes the tree for little gain, and the nearest one is the one
    # that can actually kill us this turn.
    opponents.sort(key=lambda s: abs(s.head[0] - board.get(board.me).head[0])
                   + abs(s.head[1] - board.get(board.me).head[1]))
    threat = opponents[0]
    replies = board.legal_moves(threat.sid) or [None]

    worst = float("inf")
    for reply in replies:
        moves = {board.me: my_move}
        if reply is not None:
            moves[threat.sid] = reply
        nb = board.step(moves)
        worst = min(worst, _search(nb, depth - 1, deadline))
        if time.perf_counter() > deadline:
            break
    return worst


def _search(board: SimBoard, depth: int, deadline: float) -> float:
    me = board.get(board.me)
    if me is None or not me.alive:
        return -W_DEAD
    if depth <= 0 or time.perf_counter() > deadline:
        return evaluate(board)

    moves = board.legal_moves(board.me)
    if not moves:
        return -W_DEAD

    best = -float("inf")
    for mv in moves:
        best = max(best, _worst_reply(board, mv, depth, deadline))
        if time.perf_counter() > deadline:
            break
    return best


def veto(
    payload: Mapping[str, Any],
    ranked_moves: Sequence[str],
    *,
    depth: int = 3,
    time_budget_ms: float = 90.0,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Take the policy's ranked moves and skip ones that lose within ``depth``.

    Search as a *filter*, not a driver. Driving with it scores 1.7% win rate at
    average length 5.9: the paranoid opponent model plus a horizon of 2-3 plies
    keeps it out of contested space, so it survives and starves. What it is
    genuinely good at is seeing a forced loss a couple of moves out -- exactly
    the head-to-head deaths that account for most of the policy's losses.

    So the policy keeps choosing (it eats well and grows to ~20), and this only
    rejects a choice that demonstrably dies. Returns the first surviving move,
    or None if the search could not run.
    """
    t0 = time.perf_counter()
    deadline = t0 + time_budget_ms / 1000.0
    board = board_from_payload(payload)
    if board is None:
        return None, {"reason": "unparseable board"}

    legal = set(board.legal_moves(board.me))
    if not legal:
        return None, {"reason": "no legal move"}

    candidates = [m for m in ranked_moves if m in legal] or sorted(legal)
    scores: Dict[str, float] = {}
    fatal_threshold = -W_DEAD / 2.0

    for mv in candidates:
        scores[mv] = _worst_reply(board, mv, depth, deadline)
        if scores[mv] > fatal_threshold:
            # First choice the opponents cannot force a loss against.
            return mv, {
                "picked": mv,
                "vetoed": [m for m in candidates[:candidates.index(mv)]],
                "scores": {m: round(s, 1) for m, s in scores.items()},
                "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }
        if time.perf_counter() > deadline:
            break

    # Everything we looked at loses; take the least-bad rather than the first.
    if scores:
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return best, {
            "picked": best,
            "all_losing": True,
            "scores": {m: round(s, 1) for m, s in scores.items()},
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
    return None, {"reason": "no evaluation completed"}


def choose_move(
    payload: Mapping[str, Any],
    *,
    max_depth: int = 4,
    time_budget_ms: float = 120.0,
    preferred: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Iterative-deepening search. Returns (move, debug) or (None, debug).

    Returning None lets the caller fall back; the search never guesses when it
    could not evaluate anything.
    """
    t0 = time.perf_counter()
    deadline = t0 + time_budget_ms / 1000.0
    board = board_from_payload(payload)
    if board is None:
        return None, {"reason": "unparseable board"}

    root_moves = board.legal_moves(board.me)
    if not root_moves:
        return None, {"reason": "no legal move"}
    if len(root_moves) == 1:
        return root_moves[0], {"depth": 0, "reason": "forced", "ms": 0.0}

    best_move = root_moves[0]
    best_scores: Dict[str, float] = {}
    depth_reached = 0

    for depth in range(1, max_depth + 1):
        scores: Dict[str, float] = {}
        completed = True
        for mv in root_moves:
            scores[mv] = _worst_reply(board, mv, depth, deadline)
            if time.perf_counter() > deadline:
                completed = False
                break
        if scores:
            # Keep a partial ply only if it evaluated every root move.
            if completed:
                best_scores = scores
                depth_reached = depth
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                top = ranked[0][1]
                tied = [m for m, s in ranked if s >= top - 1e-9]
                best_move = preferred if (preferred in tied) else tied[0]
        if not completed or time.perf_counter() > deadline:
            break

    return best_move, {
        "depth": depth_reached,
        "scores": {m: round(s, 1) for m, s in best_scores.items()},
        "ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }
