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
        self._env = self._make_env()
        self._your_pid = 0
        logger.info(
            "Loaded %s algorithm=%s mode=%s in_channels=%s device=%s",
            ckpt,
            self.meta.get("algorithm"),
            self.meta.get("mode"),
            self.meta.get("in_channels"),
            dev,
        )

    def _make_env(self) -> Any:
        mode = self.meta.get("mode", "restricted_standard")
        num_players = self.meta.get("num_players")
        return make_env(mode, num_players=num_players)

    def _rebuild_env_if_needed(self, num_players: int, width: int, height: int) -> None:
        cfg = self._env.cfg
        if (
            self._env.num_players == num_players
            and cfg.w == width
            and cfg.h == height
        ):
            return
        logger.warning(
            "Rebuilding env: players %s->%s board %sx%s->%sx%s",
            self._env.num_players,
            num_players,
            cfg.w,
            cfg.h,
            width,
            height,
        )
        mode = self.meta.get("mode", "restricted_standard")
        self._env.close()
        self._env = make_env(mode, num_players=num_players)
        self._env.cfg.w = width
        self._env.cfg.h = height

    def on_game_start(self, payload: Mapping[str, Any]) -> None:
        self._pid_by_snake_id = assign_player_ids(payload)
        state, your_pid = request_to_state(payload, pid_by_snake_id=self._pid_by_snake_id)
        self._your_pid = your_pid
        width, height = _board_dims(payload)
        self._rebuild_env_if_needed(len(self._pid_by_snake_id), width, height)
        self._env.set_state(state)

    def on_game_end(self, payload: Mapping[str, Any]) -> None:
        del payload  # nothing to persist for offline training in the server process

    def decide_move(self, payload: Mapping[str, Any]) -> str:
        if not self._pid_by_snake_id:
            self.on_game_start(payload)

        state, your_pid = request_to_state(payload, pid_by_snake_id=self._pid_by_snake_id)
        self._your_pid = your_pid
        width, height = _board_dims(payload)
        self._rebuild_env_if_needed(len(self._pid_by_snake_id), width, height)

        if not state.snakes_alive[your_pid]:
            return self._random_legal_or_fallback(state, your_pid)

        try:
            self._env.set_state(state)
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
            logger.exception("Inference failed; using fallback move")
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
        if self._env is not None and not self._env.is_closed:
            self._env.close()


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
