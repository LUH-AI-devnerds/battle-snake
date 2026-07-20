#!/usr/bin/env python3
"""Smoke-test Blackout API handlers using a public replay frame."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "agent" / "src"))

from battlesnake_ai.inference.runtime import SnakeRuntime  # noqa: E402


def _fetch_replay(game_id: int) -> dict:
    url = f"https://www.tnt.uni-hannover.de/bs-blackout-2026/api/replay/{game_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "battle-snake-test/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _move_to_api_payload(move: dict, *, game: dict, you_index: int = 0) -> dict:
    snakes = list(move.get("snakes") or [])
    you = snakes[you_index] if snakes else {}
    return {
        "game": game,
        "turn": int(move.get("turn", 0)),
        "board": {
            "width": move["width"],
            "height": move["height"],
            "food": move.get("food") or [],
            "hazards": move.get("hazards") or [],
            "snakes": snakes,
        },
        "you": you,
    }


def main() -> None:
    game_id = int(os.environ.get("BLACKOUT_GAME_ID", "33946"))
    ckpt = os.environ.get(
        "BATTLE_SNAKE_CHECKPOINT",
        "best_checkpoint/rainbow_v2_best.pt",
    )
    ckpt_path = _REPO / ckpt
    print(f"Loading {ckpt_path}")
    rt = SnakeRuntime(ckpt_path)
    data = _fetch_replay(game_id)
    game = data.get("game") or {}
    moves = data["moves"]
    print(f"Replay {game_id}: {len(moves)} frames, testing 20 /move calls")

    rt.on_game_start(_move_to_api_payload(moves[0], game=game))
    t0 = time.perf_counter()
    for i in range(1, min(21, len(moves))):
        payload = _move_to_api_payload(moves[i], game=game)
        mv = rt.decide_move(payload)
        assert mv in ("up", "down", "left", "right"), mv
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"OK — 20 moves, total {elapsed_ms:.1f} ms ({elapsed_ms / 20:.1f} ms/move)")
    rt.close()


if __name__ == "__main__":
    main()
