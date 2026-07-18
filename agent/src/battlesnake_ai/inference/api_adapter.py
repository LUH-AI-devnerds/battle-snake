"""
Convert Battlesnake / Blackout HTTP JSON into hisss ``BattleSnakeState``.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Mapping, Optional, Tuple

from hisss.game.state import BattleSnakeState

# hisss: UP=0, RIGHT=1, DOWN=2, LEFT=3
ACTION_NAMES = ("up", "right", "down", "left")
ACTION_FROM_NAME = {name: idx for idx, name in enumerate(ACTION_NAMES)}

# hisss 1.3.0+ added optional fields; 1.2.0 rejects them. Detect once at import.
_STATE_PARAMS = set(inspect.signature(BattleSnakeState.__init__).parameters)


def _snake_id(snake: Mapping[str, Any]) -> str:
    return str(snake.get("id", snake.get("name", "")))


def _body_coords(snake: Mapping[str, Any]) -> List[Tuple[int, int]]:
    body = snake.get("body") or []
    if not body and snake.get("head"):
        body = [snake["head"]]
    out: List[Tuple[int, int]] = []
    for seg in body:
        x, y = int(seg["x"]), int(seg["y"])
        if x < 0 or y < 0:
            continue
        out.append((x, y))
    # Battlesnake body is head-first; hisss expects the same order.
    deduped: List[Tuple[int, int]] = []
    for pt in out:
        if not deduped or deduped[-1] != pt:
            deduped.append(pt)
    return deduped


def _collect_snakes(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    board = payload.get("board")
    if board is not None:
        snakes = list(board.get("snakes") or [])
        return snakes
    # Blackout replay move shape (flat board fields).
    return list(payload.get("snakes") or []) + list(payload.get("dead_snakes") or [])


def _board_dims(payload: Mapping[str, Any]) -> Tuple[int, int]:
    board = payload.get("board")
    if board is not None:
        return int(board["width"]), int(board["height"])
    return int(payload["width"]), int(payload["height"])


def _food_list(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    board = payload.get("board")
    if board is not None:
        return list(board.get("food") or [])
    return list(payload.get("food") or [])


def assign_player_ids(payload: Mapping[str, Any]) -> Dict[str, int]:
    """Stable snake-id -> player index (sorted by id)."""
    snakes = _collect_snakes(payload)
    ids = sorted({_snake_id(s) for s in snakes if _snake_id(s)})
    you_id = _snake_id(payload["you"]) if payload.get("you") else None
    if you_id and you_id not in ids:
        ids.append(you_id)
        ids.sort()
    return {sid: pid for pid, sid in enumerate(ids)}


def _ghost_corner(pid: int, width: int, height: int) -> List[Tuple[int, int]]:
    """Far-away placeholder body so FOW-hidden snakes are not treated as dead."""
    corners = [
        (1, 1),
        (width - 2, 1),
        (1, height - 2),
        (width - 2, height - 2),
    ]
    x, y = corners[int(pid) % 4]
    return [(x, y), (x, y), (x, y)]


def request_to_state(
    payload: Mapping[str, Any],
    *,
    pid_by_snake_id: Optional[Mapping[str, int]] = None,
    ghosts: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Tuple[BattleSnakeState, int]:
    """
    Build ``BattleSnakeState`` and return ``(state, your_player_index)``.

    ``payload`` is a Blackout ``/start`` or ``/move`` body (``game``, ``turn``, ``board``, ``you``).

    Under fog-of-war, unseen opponents are omitted from ``board.snakes``. Marking
    them dead makes hisss treat the game as terminal (sole survivor) and
    ``get_obs`` fails — which previously fell back to always-``up``. Missing
    snakes are kept alive via ``ghosts`` / corner placeholders instead.
    """
    if "you" not in payload:
        raise ValueError("Request body must include 'you'")

    pid_map = dict(pid_by_snake_id or assign_player_ids(payload))
    you_id = _snake_id(payload["you"])
    if you_id not in pid_map:
        pid_map[you_id] = len(pid_map)
    your_pid = pid_map[you_id]
    num_players = max(max(pid_map.values()) + 1, 4)

    turn = int(payload.get("turn") or 0)
    width, height = _board_dims(payload)
    snakes = _collect_snakes(payload)
    # Also consider dead_snakes so we permanently retire eliminated opponents.
    board = payload.get("board") or {}
    dead = list(board.get("dead_snakes") or payload.get("dead_snakes") or [])
    dead_ids = {_snake_id(s) for s in dead if _snake_id(s)}

    snake_pos: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(num_players)}
    snakes_alive = [False] * num_players
    snake_health = [0] * num_players
    snake_len = [0] * num_players
    seen: set[int] = set()

    for snake in list(snakes) + list(dead):
        sid = _snake_id(snake)
        pid = pid_map.get(sid)
        if pid is None:
            continue
        coords = _body_coords(snake)
        snake_pos[pid] = coords or _ghost_corner(pid, width, height)
        health = int(snake.get("health", 0))
        snake_health[pid] = health
        snake_len[pid] = int(snake.get("length", len(coords) or 3))
        elim = snake.get("elimination") or snake.get("elimination_event")
        # Dead-list entries or explicit elimination → dead; otherwise alive.
        snakes_alive[pid] = sid not in dead_ids and health > 0 and elim is None
        seen.add(pid)

    ghosts = ghosts or {}
    for pid in range(num_players):
        if pid in seen:
            continue
        if pid == your_pid:
            you = payload.get("you")
            snake_pos[pid] = _body_coords(you) if you else _ghost_corner(pid, width, height)
            snake_health[pid] = int((you or {}).get("health", 100))
            snake_len[pid] = int((you or {}).get("length", 3))
            snakes_alive[pid] = True
            continue
        ghost = ghosts.get(pid)
        if ghost and ghost.get("alive", True):
            snake_pos[pid] = list(ghost.get("pos") or _ghost_corner(pid, width, height))
            snake_health[pid] = int(ghost.get("health", 100))
            snake_len[pid] = int(ghost.get("length", 3))
            snakes_alive[pid] = True
        else:
            # Never-seen FOW opponent: keep alive at a far corner so the match
            # is not terminal and the 4-player observation stays valid.
            snake_pos[pid] = _ghost_corner(pid, width, height)
            snake_health[pid] = 100
            snake_len[pid] = 3
            snakes_alive[pid] = True

    # Avoid overlapping our head with a ghost corner on the same cell.
    you_head = snake_pos[your_pid][0] if snake_pos.get(your_pid) else None
    if you_head is not None:
        for pid in range(num_players):
            if pid == your_pid or not snakes_alive[pid]:
                continue
            if snake_pos[pid] and snake_pos[pid][0] == you_head:
                snake_pos[pid] = _ghost_corner((pid + 1) % 4, width, height)

    food_pos: List[List[int]] = []
    food_spawn_turns: List[int] = []
    for fp in _food_list(payload):
        x, y = int(fp["x"]), int(fp["y"])
        if x < 0 or y < 0:
            continue
        food_pos.append([x, y])
        food_spawn_turns.append(int(fp.get("spawn_turn", turn)))

    kwargs: Dict[str, Any] = {
        "turn": turn,
        "snakes_alive": snakes_alive,
        "snake_pos": snake_pos,
        "food_pos": food_pos,
        "snake_health": snake_health,
        "snake_len": snake_len,
    }
    if "food_spawn_turns" in _STATE_PARAMS:
        kwargs["food_spawn_turns"] = food_spawn_turns
    if "elimination_events" in _STATE_PARAMS:
        kwargs["elimination_events"] = None

    state = BattleSnakeState(**kwargs)
    return state, your_pid


def action_index_to_move(action: int) -> str:
    if action < 0 or action >= len(ACTION_NAMES):
        raise ValueError(f"Invalid action index: {action}")
    return ACTION_NAMES[action]
