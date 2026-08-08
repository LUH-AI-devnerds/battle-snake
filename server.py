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
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Mapping

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
from battlesnake_ai.inference.telemetry import Telemetry  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("battlesnake.server")
# Separate logger so a competition run can be filtered to just the move trace
# with `grep MOVE` / `grep GAME` in the Railway log viewer.
move_logger = logging.getLogger("battlesnake.move")

SNAKE_AUTHOR = os.environ.get("SNAKE_AUTHOR", "Battle Snake")
SNAKE_COLOR = os.environ.get("SNAKE_COLOR", "#4488ff")
# Per-move logging is on by default: during a leaderboard run the move trace is
# the only record of what the snake actually did. Set MOVE_LOG=0 to quieten it.
MOVE_LOG = os.environ.get("MOVE_LOG", "1").strip().lower() not in {"0", "false", "no", "off"}

_runtime: SnakeRuntime | None = None
_telemetry = Telemetry()


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


def _game_id(payload: Mapping[str, Any]) -> str:
    return str((payload.get("game") or {}).get("id") or "unknown")


@app.post("/start")
def start_game(body: BattlesnakeRequest) -> Dict[str, str]:
    # Like /move, this must never raise. The engine does not need anything from
    # the body, and decide_move re-initialises the game on its first call if
    # this failed, so swallowing here costs nothing and a 500 might not.
    try:
        payload = body.model_dump()
        gid = _game_id(payload)
        if _runtime is not None:
            _runtime.on_game_start(payload)
        _telemetry.on_start(gid)
        board = payload.get("board") or {}
        logger.info(
            "GAME START g=%s turn=%s board=%sx%s snakes=%s",
            gid, body.turn, board.get("width"), board.get("height"),
            len(board.get("snakes") or []),
        )
    except Exception:
        logger.exception("/start failed; first /move will re-initialise the game")
    return {}


def _last_resort_move(payload: Mapping[str, Any]) -> str:
    """A legal-looking move derived without touching anything that can throw."""
    try:
        from battlesnake_ai.inference.safe_move import legal_moves

        options = legal_moves(payload)
        if options:
            return options[0]
    except Exception:
        pass
    return os.environ.get("FALLBACK_MOVE", "up")


@app.post("/move")
def move(body: BattlesnakeRequest) -> Dict[str, str]:
    assert _runtime is not None
    payload = body.model_dump()
    gid = _game_id(payload)
    you = payload.get("you") or {}

    # /move must never raise. An exception here becomes a 500, and the engine
    # treats a failed response as a forfeited move -- strictly worse than any
    # legal move we could have returned. decide_move guards the model path
    # internally, but the surrounding code (and choose_safe_move itself) can
    # still throw on a malformed board, which is how a bad payload turned into
    # a 500 with the telemetry recording nothing at all.
    t_handler = time.perf_counter()
    try:
        direction = _runtime.decide_move(payload)
        decision = _runtime.last_decision()
    except Exception:
        direction = _last_resort_move(payload)
        decision = {
            "move": direction,
            "source": "handler_exception",
            "ms": round((time.perf_counter() - t_handler) * 1000.0, 1),
            "turn": payload.get("turn"),
            "legal": [],
        }
        logger.exception(
            "MOVE HANDLER CRASHED g=%s turn=%s -- returned %s so the move is not "
            "forfeited, but the snake is playing blind until this is fixed",
            gid[:8], payload.get("turn"), direction,
        )

    _telemetry.on_move(gid, decision, you)

    if MOVE_LOG:
        source = str(decision.get("source") or "?")
        ms = float(decision.get("ms") or 0.0)
        # One compact line per move. Fields are key=value so a competition run
        # can be sliced with grep/awk straight out of the Railway log viewer.
        # "!" marks a move the RL path did not produce, or one that ran long.
        flags = ""
        if "exception" in source:
            flags += " !FALLBACK"
        if ms > 500.0:
            flags += " !OVER_BUDGET"
        elif ms > 250.0:
            flags += " !SLOW"
        move_logger.info(
            "MOVE g=%s t=%s mv=%s src=%s ms=%.1f len=%s hp=%s legal=%s%s",
            gid[:8], decision.get("turn"), direction, source, ms,
            you.get("length"), you.get("health"),
            ",".join(decision.get("legal") or []), flags,
        )
    return {"move": direction}


@app.post("/end")
def end_game(body: BattlesnakeRequest) -> Dict[str, str]:
    try:
        _end_game_inner(body)
    except Exception:
        logger.exception("/end failed (game bookkeeping only; play is unaffected)")
    return {}


def _end_game_inner(body: BattlesnakeRequest) -> None:
    payload = body.model_dump()
    if _runtime is not None:
        _runtime.on_game_end(payload)
    gid = _game_id(payload)
    summary = _telemetry.on_end(gid, payload)
    if summary:
        # The single most useful line in the log during a leaderboard run:
        # how the game ended, and whether the policy was actually playing it.
        logger.info(
            "GAME END g=%s outcome=%s turns=%s len=%s snakes_left=%s "
            "fallbacks=%s p50=%sms p95=%sms max=%sms",
            gid[:8], summary["outcome"], summary["turns"], summary["our_length"],
            summary["snakes_at_end"], summary["fallbacks"],
            summary["latency_ms"]["p50"], summary["latency_ms"]["p95"],
            summary["latency_ms"]["max"],
        )
        if summary["fallbacks"]:
            logger.error(
                "GAME END g=%s served %s/%s moves from the last-resort heuristic "
                "-- the RL agent was not playing this game",
                gid[:8], summary["fallbacks"], summary["turns"],
            )
    else:
        logger.info("GAME END g=%s (no recorded moves)", gid[:8])


@app.get("/stats")
def stats(recent_games: int = 10) -> Dict[str, Any]:
    """Live match telemetry. The competition server has no shell, so this is
    the way to check what the snake is actually doing mid-leaderboard-run."""
    snap = _telemetry.snapshot(recent_games=max(0, min(recent_games, 60)))
    snap["healthy"] = _telemetry.healthy()
    snap["checkpoint"] = os.environ.get("BATTLE_SNAKE_CHECKPOINT", "")
    snap["strategy"] = os.environ.get("MOVE_STRATEGY", "model")
    return snap


@app.get("/stats/moves")
def recent_moves(n: int = 50) -> Dict[str, Any]:
    """Most recent per-move decisions, newest first."""
    return {"moves": _telemetry.recent_decisions(max(1, min(n, 300)))}
