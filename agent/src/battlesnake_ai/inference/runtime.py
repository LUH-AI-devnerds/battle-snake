"""
Load a checkpoint once and answer Blackout ``/move`` requests via hisss + RL policy.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.inference.agent_loader import load_agent
from battlesnake_ai.inference.api_adapter import (
    ACTION_FROM_NAME,
    _board_dims,
    action_index_to_move,
    assign_player_ids,
    request_to_state,
)
from battlesnake_ai.models.dqn import DQN
from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.models.rainbow_dqn import RainbowDQN
from battlesnake_ai.training.action_selection import masked_argmax

logger = logging.getLogger(__name__)


def _resolve_checkpoint_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        latest = p / "rainbow_latest.pt"
        if latest.is_file():
            return latest
        pts = sorted(p.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if pts:
            return pts[0]
    raise FileNotFoundError(f"No checkpoint at {path}")


class SnakeRuntime:
    """Holds env + model; syncs game state from Battlesnake JSON each turn."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: Optional[str] = None,
        fallback_move: str = "up",
    ) -> None:
        ckpt = _resolve_checkpoint_path(checkpoint)
        dev = torch.device(device or os.environ.get("TORCH_DEVICE", "cpu"))
        self.device = dev
        self.model, self.meta = load_agent(ckpt, device=dev)
        self.fallback_move = fallback_move if fallback_move in ACTION_FROM_NAME else "up"
        self._pid_by_snake_id: Dict[str, int] = {}
        self._fallback_count = 0
        # Cache envs by (num_players, width, height). hisss envs are never closed
        # and recreated at runtime: doing so double-frees native memory and crashes
        # the whole process. For Blackout only one config (4, 15, 15) is ever used.
        self._env_cache: Dict[Tuple[int, int, int], Any] = {}
        self._env = self._make_env()
        self._env_cache[(self._env.num_players, self._env.cfg.w, self._env.cfg.h)] = self._env
        self._your_pid = 0
        hisss_ver = "unknown"
        try:
            import importlib.metadata as _imd

            hisss_ver = _imd.version("hisss")
        except Exception:
            pass
        logger.info(
            "Loaded %s algorithm=%s mode=%s in_channels=%s device=%s hisss=%s fallback=%s",
            ckpt,
            self.meta.get("algorithm"),
            self.meta.get("mode"),
            self.meta.get("in_channels"),
            dev,
            hisss_ver,
            self.fallback_move,
        )

    def _make_env(self) -> Any:
        mode = self.meta.get("mode", "restricted_standard")
        num_players = self.meta.get("num_players")
        return make_env(mode, num_players=num_players)

    def _select_env(self, num_players: int, width: int, height: int) -> Any:
        """Return a cached env for this config, building it once if needed.

        Never closes/recreates an existing env: the native hisss close+rebuild
        cycle double-frees memory and aborts the process. Envs are cached and
        kept alive for the lifetime of the runtime instead.
        """
        num_players = max(1, int(num_players))
        key = (num_players, width, height)
        env = self._env_cache.get(key)
        if env is None:
            if (self._env.num_players, self._env.cfg.w, self._env.cfg.h) != key:
                logger.warning(
                    "Building env for players=%s board=%sx%s (current %s players %sx%s)",
                    num_players,
                    width,
                    height,
                    self._env.num_players,
                    self._env.cfg.w,
                    self._env.cfg.h,
                )
            mode = self.meta.get("mode", "restricted_standard")
            env = make_env(mode, num_players=num_players)
            if env.cfg.w != width or env.cfg.h != height:
                env.cfg.w = width
                env.cfg.h = height
            self._env_cache[key] = env
        self._env = env
        return env

    def on_game_start(self, payload: Mapping[str, Any]) -> None:
        try:
            self._pid_by_snake_id = assign_player_ids(payload)
            if not self._pid_by_snake_id:
                return
            state, your_pid = request_to_state(payload, pid_by_snake_id=self._pid_by_snake_id)
            self._your_pid = your_pid
            width, height = _board_dims(payload)
            self._select_env(len(self._pid_by_snake_id), width, height)
            self._env.set_state(state)
        except Exception:
            logger.exception("on_game_start failed; will re-initialize on first move")

    def on_game_end(self, payload: Mapping[str, Any]) -> None:
        del payload  # nothing to persist for offline training in the server process

    def decide_move(self, payload: Mapping[str, Any]) -> str:
        try:
            if not self._pid_by_snake_id:
                self.on_game_start(payload)
            if not self._pid_by_snake_id:
                return self.fallback_move

            state, your_pid = request_to_state(payload, pid_by_snake_id=self._pid_by_snake_id)
            self._your_pid = your_pid
            width, height = _board_dims(payload)
            self._select_env(len(self._pid_by_snake_id), width, height)

            if not state.snakes_alive[your_pid]:
                return self._random_legal_or_fallback(state, your_pid)

            self._env.set_state(state)
            # hisss raises on get_obs() once the game is already over (e.g. we
            # received a /move after the board reached a terminal state).
            if self._env.is_terminal():
                return self._random_legal_or_fallback(state, your_pid)

            obs, _, _ = self._env.get_obs()
            players_here = list(self._env.players_at_turn())
            if your_pid not in players_here:
                return self._random_legal_or_fallback(state, your_pid)

            row_idx = players_here.index(your_pid)
            sl = obs[row_idx : row_idx + 1]
            action = self._select_action(sl, your_pid)
            la = self._env.available_actions(your_pid)
            if action not in la:
                action = int(random.choice(la)) if la else 0
            return action_index_to_move(action)
        except Exception:
            self._fallback_count += 1
            logger.exception(
                "Inference failed; using fallback move=%s (fallback_count=%s). "
                "If this happens every turn the snake will walk in one direction.",
                self.fallback_move,
                self._fallback_count,
            )
            return self.fallback_move

    def _select_action(self, obs_slice: np.ndarray, pid: int) -> int:
        la = self._env.available_actions(pid)
        if isinstance(self.model, (DQN, RainbowDQN)):
            with torch.no_grad():
                q = self.model(obs_slice).detach().cpu().numpy()[0]
            return masked_argmax(q, la)
        if isinstance(self.model, PPOPolicy):
            with torch.no_grad():
                logits = self.model.actor_logits(obs_slice)[0].detach().cpu().numpy()
            mask = np.full(logits.shape, -1e9, dtype=np.float32)
            for a in la:
                mask[a] = logits[a]
            return int(mask.argmax())
        with torch.no_grad():
            logits = self.model(obs_slice).detach().cpu().numpy()[0]
        return masked_argmax(logits, la)

    def _random_legal_or_fallback(self, state: Any, your_pid: int) -> str:
        try:
            self._env.set_state(state)
            la = self._env.available_actions(your_pid)
            if la:
                return action_index_to_move(int(random.choice(la)))
        except Exception:
            pass
        return self.fallback_move

    def close(self) -> None:
        for env in self._env_cache.values():
            try:
                if env is not None and not env.is_closed:
                    env.close()
            except Exception:
                logger.exception("Error closing env during shutdown")
        self._env_cache.clear()


def default_checkpoint_from_env() -> Path:
    raw = os.environ.get(
        "BATTLE_SNAKE_CHECKPOINT",
        "logs/checkpoints/rainbow_20260602_182838_ep75.pt",
    )
    root = Path(__file__).resolve().parents[4]
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    return _resolve_checkpoint_path(p)
