"""
Battlesnake Blackout 2026 competition server.

API: https://www.tnt.uni-hannover.de/bs-blackout-2026/doc

Run locally:
  export PYTHONPATH=agent/src
  export BATTLE_SNAKE_CHECKPOINT=logs/checkpoints/rainbow_20260602_182838_ep75.pt
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parent
_AGENT_SRC = _REPO_ROOT / "agent" / "src"
if str(_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(_AGENT_SRC))

# Must run before hisss is used (view-radius row index breaks after eliminations).
from battlesnake_ai.env.hisss_view_radius_fix import apply_view_radius_row_index_fix  # noqa: E402

if not apply_view_radius_row_index_fix():
    raise RuntimeError(
        "hisss view-radius patch failed — /move would fall back to FALLBACK_MOVE after eliminations"
    )

from battlesnake_ai.inference.runtime import SnakeRuntime, default_checkpoint_from_env  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("battlesnake.server")

SNAKE_AUTHOR = os.environ.get("SNAKE_AUTHOR", "Battle Snake")
SNAKE_COLOR = os.environ.get("SNAKE_COLOR", "#4488ff")

_runtime: SnakeRuntime | None = None


class BattlesnakeRequest(BaseModel):
    game: Dict[str, Any] = Field(default_factory=dict)
    turn: int = 0
    board: Dict[str, Any] = Field(default_factory=dict)
    you: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runtime
    ckpt = os.environ.get("BATTLE_SNAKE_CHECKPOINT")
    path = Path(ckpt) if ckpt else default_checkpoint_from_env()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    _runtime = SnakeRuntime(
        path,
        device=os.environ.get("TORCH_DEVICE"),
        fallback_move=os.environ.get("FALLBACK_MOVE", "up"),
    )
    logger.info("Snake server ready | author=%s | checkpoint=%s", SNAKE_AUTHOR, path)
    yield
    if _runtime is not None:
        _runtime.close()
        _runtime = None


app = FastAPI(title="Battle Snake Blackout", lifespan=lifespan)


@app.get("/")
def snake_info() -> Dict[str, str]:
    """Snake metadata (author must match registration when set)."""
    return {"author": SNAKE_AUTHOR, "color": SNAKE_COLOR}


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness + patch/checkpoint diagnostics for deploy smoke tests."""
    import importlib.metadata as im

    hisss_ver = "unknown"
    try:
        hisss_ver = im.version("hisss")
    except Exception:
        pass
    patch_ok = False
    try:
        import hisss.game.battlesnake as bsm

        patch_ok = bool(getattr(bsm.BattleSnakeGame, "_bs_ai_view_radius_row_fix", False))
    except Exception:
        pass
    ckpt = os.environ.get("BATTLE_SNAKE_CHECKPOINT", "")
    return {
        "status": "ok" if _runtime is not None and patch_ok else "degraded",
        "hisss": hisss_ver,
        "view_radius_patch": patch_ok,
        "checkpoint": ckpt,
        "survival_filter": os.environ.get("SURVIVAL_FILTER", "1"),
    }


@app.post("/start")
def start_game(body: BattlesnakeRequest) -> Dict[str, str]:
    assert _runtime is not None
    payload = body.model_dump()
    _runtime.on_game_start(payload)
    logger.info("Game start id=%s turn=%s", payload.get("game", {}).get("id"), body.turn)
    return {}


@app.post("/move")
def move(body: BattlesnakeRequest) -> Dict[str, str]:
    assert _runtime is not None
    payload = body.model_dump()
    direction = _runtime.decide_move(payload)
    return {"move": direction}


@app.post("/end")
def end_game(body: BattlesnakeRequest) -> Dict[str, str]:
    assert _runtime is not None
    _runtime.on_game_end(body.model_dump())
    logger.info("Game end id=%s", body.game.get("id"))
    return {}
