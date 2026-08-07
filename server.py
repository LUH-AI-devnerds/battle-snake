"""
Battlesnake Blackout 2026 competition server.

API: https://www.tnt.uni-hannover.de/bs-blackout-2026/doc

Run locally:
  export PYTHONPATH=agent/src
  export BATTLE_SNAKE_CHECKPOINT=logs/checkpoints/rainbow_20260602_182838_ep75.pt
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os

# Must be set before torch's OpenMP/MKL runtime initializes (first import
# anywhere in the process), so this has to come before every other import.
# Containers like Railway report the host's full CPU count via os.cpu_count()
# while granting a small CPU quota. PyTorch's default thread pool sizes itself
# to that reported count, so on such a host it spawns dozens of threads that
# contend for a sliver of real CPU -- for a single small forward pass per
# request that overhead dominates. Measured: 12.8 ms/move locally (128 cores,
# unconstrained) vs 460-1000 ms/move on Railway with default threading,
# against a 500 ms hard budget. Pinning to 1 thread fixes it because there is
# no parallelism to exploit in a batch-of-one 15x15x17 conv forward pass.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

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

    @field_validator("turn", mode="before")
    @classmethod
    def _coerce_turn(cls, value: Any) -> int:
        if value is None:
            return 0
        return int(value)


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
    last = _runtime.last_decision() if _runtime is not None else {}

    # A crash in the model path is caught and answered with the last-resort
    # heuristic, so /move keeps returning 200 and nothing looks wrong from
    # outside. That is exactly how a null-health TypeError went unnoticed
    # while the RL agent sat idle for thousands of live moves. Any fallback at
    # all is a real defect: surface it here instead of hiding it.
    fallbacks = int(last.get("fallback_count") or 0) if last else 0
    rl_driving = str(last.get("source") or "").startswith(("model", "tactics"))
    if _runtime is None or not patch_ok:
        status = "degraded"
    elif fallbacks > 0:
        status = "degraded"
    elif last and not rl_driving:
        status = "degraded"
    else:
        status = "ok"

    return {
        "status": status,
        "hisss": hisss_ver,
        "view_radius_patch": patch_ok,
        "checkpoint": ckpt,
        "survival_filter": os.environ.get("SURVIVAL_FILTER", "0"),
        # Non-zero means the RL path threw and the crude heuristic answered.
        # Should always be 0 in a healthy deployment.
        "fallback_count": fallbacks,
        "rl_driving": rl_driving,
        "last_decision": last,
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
