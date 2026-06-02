"""
Convert Battlesnake Blackout replay JSON into board state for the local matplotlib GUI.

Blackout uses standard Battlesnake coordinates (x right, y up from bottom-left).
hisss / ``state_to_rgb_board`` use (row, col) with row 0 at the top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

# pid -> display name (from leaderboard or game page)
SnakeLabels = Dict[int, str]


def bs_xy_to_row_col(x: int, y: int, height: int) -> Tuple[int, int]:
    return height - 1 - int(y), int(x)


def move_to_board_state(
    move: Mapping[str, Any],
    *,
    pid_by_snake_id: Mapping[str, int],
) -> Tuple[int, int, Dict[int, List[Tuple[int, int]]], List[Tuple[int, int]], List[bool]]:
    """
    Returns ``(width, height, snake_positions, food_positions, alive)``.
    """
    height = int(move["height"])
    width = int(move["width"])

    snake_positions: Dict[int, List[Tuple[int, int]]] = {}
    alive: List[bool] = []
    max_pid = max(pid_by_snake_id.values()) if pid_by_snake_id else -1
    n_slots = max_pid + 1

    def add_snake(snake: Mapping[str, Any], is_dead: bool) -> None:
        sid = str(snake.get("id", snake.get("name", "")))
        pid = pid_by_snake_id.get(sid)
        if pid is None:
            return
        chain: List[Tuple[int, int]] = []
        for seg in snake.get("body") or []:
            chain.append(bs_xy_to_row_col(seg["x"], seg["y"], height))
        if not chain and snake.get("head"):
            h = snake["head"]
            chain.append(bs_xy_to_row_col(h["x"], h["y"], height))
        snake_positions[pid] = chain
        while len(alive) <= pid:
            alive.append(True)
        alive[pid] = not is_dead

    for snake in move.get("snakes") or []:
        add_snake(snake, is_dead=False)
    for snake in move.get("dead_snakes") or []:
        add_snake(snake, is_dead=True)

    food_positions: List[Tuple[int, int]] = []
    for fp in move.get("food") or []:
        food_positions.append(bs_xy_to_row_col(fp["x"], fp["y"], height))

    if n_slots > 0:
        while len(alive) < n_slots:
            alive.append(False)

    return width, height, snake_positions, food_positions, alive


@dataclass
class BlackoutReplay:
    game_id: int
    raw: Dict[str, Any]
    pid_by_snake_id: Dict[str, int]
    snake_labels: SnakeLabels
    total_turns: int

    @classmethod
    def from_api(cls, data: Dict[str, Any], *, game_id: int, snake_labels: Optional[SnakeLabels] = None) -> "BlackoutReplay":
        moves = data.get("moves") or []
        if not moves:
            raise ValueError(f"Replay {game_id} has no moves")

        first = moves[0]
        all_snakes = list(first.get("snakes") or []) + list(first.get("dead_snakes") or [])
        pid_by_snake_id: Dict[str, int] = {}
        labels: SnakeLabels = {}
        for pid, snake in enumerate(all_snakes):
            sid = str(snake.get("id", snake.get("name", pid)))
            pid_by_snake_id[sid] = pid
            name = str(snake.get("name", sid))
            labels[pid] = (snake_labels or {}).get(pid, name)

        if snake_labels:
            labels.update(snake_labels)

        return cls(
            game_id=game_id,
            raw=data,
            pid_by_snake_id=pid_by_snake_id,
            snake_labels=labels,
            total_turns=int(data.get("total_turns", len(moves) - 1)),
        )

    def frame(self, turn: int) -> Tuple[int, int, Dict[int, List[Tuple[int, int]]], List[Tuple[int, int]], List[bool]]:
        moves = self.raw["moves"]
        turn = max(0, min(turn, len(moves) - 1))
        return move_to_board_state(moves[turn], pid_by_snake_id=self.pid_by_snake_id)

    def snake_info_at(self, turn: int) -> List[Dict[str, Any]]:
        move = self.raw["moves"][max(0, min(turn, len(self.raw["moves"]) - 1))]
        rows: List[Dict[str, Any]] = []
        for snake in (move.get("snakes") or []) + (move.get("dead_snakes") or []):
            sid = str(snake.get("id", ""))
            pid = self.pid_by_snake_id.get(sid)
            rows.append(
                {
                    "pid": pid,
                    "label": self.snake_labels.get(pid, snake.get("name", "?")),
                    "health": snake.get("health"),
                    "elimination": snake.get("elimination_event"),
                }
            )
        return rows
