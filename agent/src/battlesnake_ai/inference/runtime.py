"""
Load a checkpoint once and answer Blackout ``/move`` requests via hisss + RL policy.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.inference.agent_loader import load_agent
from battlesnake_ai.inference.api_adapter import (
    ACTION_FROM_NAME,
    ACTION_NAMES,
    _board_dims,
    action_index_to_move,
    assign_player_ids,
    request_to_state,
)
from battlesnake_ai.inference.safe_move import choose_safe_move, legal_moves
from battlesnake_ai.inference.survival import select_survival_action
from battlesnake_ai.inference import search, tactics
from battlesnake_ai.models.dqn import DQN
from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.models.rainbow_dqn import RainbowDQN
from battlesnake_ai.training.action_selection import masked_argmax

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


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
        # How the final move is chosen:
        #   model   — policy ranks, one-step head-to-head filter (default, shipped)
        #   veto    — model plus a lookahead search that rejects losing moves
        #   tactics — flood-fill/food/H2H search decides, model breaks ties
        #   safe    — legacy space heuristic, model breaks ties
        #
        # The default must match what is deployed. It briefly did not: the code
        # defaulted to "veto" while the Dockerfile pinned "model", so any host
        # without that env silently ran a configuration that measured no better
        # under fog-of-war visibility and spent ~40% of the real latency budget.
        self.move_strategy = os.environ.get("MOVE_STRATEGY", "model").strip().lower()
        # Time the lookahead veto may spend per move. Blackout allows 500 ms and
        # the policy itself costs ~15 ms, so this stays far inside the budget.
        self.search_budget_ms = float(os.environ.get("SEARCH_BUDGET_MS", "70"))
        # Soft combat ranking (optional). Hard safe_move filter is always on.
        self.survival_filter = _env_flag("SURVIVAL_FILTER", False)
        self.hunger_health = int(os.environ.get("SURVIVAL_HUNGER_HEALTH", "35"))
        self.combat_strategy = os.environ.get("SURVIVAL_STRATEGY", "aggressive").strip().lower()
        self._pid_by_snake_id: Dict[str, int] = {}
        self._ghosts: Dict[int, Dict[str, Any]] = {}
        self._current_game_id: Optional[str] = None
        self._fallback_count = 0
        self._last_decision: Dict[str, Any] = {}
        self._model_ranking: List[str] = []
        self._terminal_state = False
        # hisss wraps a native game object and is not thread-safe. FastAPI runs
        # these sync handlers in a thread pool, so two overlapping /move calls
        # mutate the same env via set_state/get_obs and corrupt its memory --
        # observed as "double free or corruption (out)" and a core dump at
        # concurrency 4, which takes the entire server down mid-tournament.
        # The mutable per-turn state here (_env, _ghosts, _pid_by_snake_id,
        # _model_ranking) is shared for the same reason, so one lock covers the
        # whole decision. A move costs ~15 ms against a 500 ms budget, so
        # serialising overlapping requests is essentially free.
        self._lock = threading.RLock()
        # Blackout is always 4 snakes on 15x15 even when FOW hides opponents.
        self._force_players = int(os.environ.get("FORCE_NUM_PLAYERS", "4"))
        self._env_cache: Dict[Tuple[int, int, int], Any] = {}
        self._env = self._make_env(self._force_players)
        self._env_cache[(self._env.num_players, self._env.cfg.w, self._env.cfg.h)] = self._env
        self._your_pid = 0
        hisss_ver = "unknown"
        try:
            import importlib.metadata as _imd

            hisss_ver = _imd.version("hisss")
        except Exception:
            pass
        self._warmup()
        logger.info(
            "Loaded %s algorithm=%s mode=%s in_channels=%s device=%s hisss=%s "
            "fallback=%s survival_filter=%s hunger_health=%s strategy=%s force_players=%s",
            ckpt,
            self.meta.get("algorithm"),
            self.meta.get("mode"),
            self.meta.get("in_channels"),
            dev,
            hisss_ver,
            self.fallback_move,
            self.survival_filter,
            self.hunger_health,
            self.combat_strategy,
            self._force_players,
        )

    def _warmup(self) -> None:
        """Prime torch/hisss so the first live /move stays under the 500ms budget."""
        try:
            reset_out = self._env.reset()
            if reset_out is None:
                # Some hisss builds mutate in place and return None.
                obs, _, _ = self._env.get_obs()
            else:
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            if obs is not None and len(obs) > 0:
                with torch.no_grad():
                    _ = self.model(obs[0:1])
        except Exception:
            logger.exception("Warmup failed (non-fatal)")

    def _update_ghosts(self, state: Any) -> None:
        for pid, alive in enumerate(state.snakes_alive):
            pos = list(state.snake_pos.get(pid) or [])
            self._ghosts[pid] = {
                "alive": bool(alive),
                "pos": pos,
                "health": int(state.snake_health[pid]),
                "length": int(state.snake_len[pid]),
            }

    def _make_env(self, num_players: Optional[int] = None) -> Any:
        mode = self.meta.get("mode", "restricted_standard")
        n = num_players if num_players is not None else self.meta.get("num_players")
        if n is None:
            n = self._force_players
        return make_env(mode, num_players=int(n))

    def _select_env(self, num_players: int, width: int, height: int) -> Any:
        """Return a cached env for this config, building it once if needed."""
        # Never shrink below the Blackout player count when FOW hides snakes.
        num_players = max(self._force_players, int(num_players))
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
            env = self._make_env(num_players)
            if env.cfg.w != width or env.cfg.h != height:
                env.cfg.w = width
                env.cfg.h = height
            self._env_cache[key] = env
        self._env = env
        return env

    def _merge_player_ids(self, payload: Mapping[str, Any]) -> None:
        """Keep a stable id→pid map; add newly visible FOW snakes into free slots."""
        incoming = assign_player_ids(payload)
        if not self._pid_by_snake_id:
            self._pid_by_snake_id = dict(incoming)
        else:
            used = set(self._pid_by_snake_id.values())
            for sid in incoming:
                if sid in self._pid_by_snake_id:
                    continue
                pid = next((i for i in range(self._force_players) if i not in used), None)
                if pid is None:
                    pid = len(self._pid_by_snake_id)
                self._pid_by_snake_id[sid] = pid
                used.add(pid)
        # Pad with placeholders so hisss always sees 4 player slots.
        used = set(self._pid_by_snake_id.values())
        for i in range(self._force_players):
            if i not in used:
                self._pid_by_snake_id[f"__missing_{i}"] = i
                used.add(i)

    def on_game_start(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self._on_game_start_locked(payload)

    def _on_game_start_locked(self, payload: Mapping[str, Any]) -> None:
        try:
            self._current_game_id = str((payload.get("game") or {}).get("id", "")) or None
            self._pid_by_snake_id = {}
            self._merge_player_ids(payload)
            if not self._pid_by_snake_id:
                return
            state, your_pid = request_to_state(
                payload,
                pid_by_snake_id=self._pid_by_snake_id,
                ghosts=self._ghosts,
            )
            self._your_pid = your_pid
            self._update_ghosts(state)
            width, height = _board_dims(payload)
            self._select_env(self._force_players, width, height)
            self._env.set_state(state)
        except Exception:
            logger.exception("on_game_start failed; will re-initialize on first move")

    def on_game_end(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self._pid_by_snake_id = {}
            self._ghosts = {}
            self._current_game_id = None
        del payload

    def _ensure_game(self, payload: Mapping[str, Any]) -> None:
        gid = str((payload.get("game") or {}).get("id", "")) or None
        if gid != self._current_game_id or not self._pid_by_snake_id:
            self._on_game_start_locked(payload)
        else:
            self._merge_player_ids(payload)

    def decide_move(self, payload: Mapping[str, Any]) -> str:
        t0 = time.perf_counter()
        preferred: Optional[str] = None
        source = "safe"
        ranking: List[str] = []
        self._terminal_state = False
        t_ensure = t_model = t_strategy = 0.0
        try:
            # Only the env-touching part is serialised. hisss is the piece that
            # is not thread-safe; the search and heuristics below are pure
            # Python over the JSON payload, so holding the lock through them
            # would queue every concurrent game behind one search budget
            # (8 games x 70 ms = 560 ms, past the 500 ms deadline) for no reason.
            with self._lock:
                t1 = time.perf_counter()
                self._ensure_game(payload)
                t_ensure = (time.perf_counter() - t1) * 1000.0
                if self._pid_by_snake_id:
                    t1 = time.perf_counter()
                    self._model_ranking = []
                    preferred = self._model_move(payload)
                    ranking = list(self._model_ranking)
                    t_model = (time.perf_counter() - t1) * 1000.0
                    if preferred is not None:
                        source = "model"
        except Exception:
            self._fallback_count += 1
            # This is never routine. When it fires the RL agent is not playing
            # at all -- the crude space/food heuristic is. Log the first one
            # unmistakably, then throttle so a persistent fault cannot bury the
            # rest of the log at ~3 moves/second.
            if self._fallback_count == 1 or self._fallback_count % 50 == 0:
                logger.exception(
                    "RL PATH DOWN -- serving last-resort heuristic instead of the policy "
                    "(fallback_count=%s, game=%s, turn=%s). The snake is playing badly "
                    "until this is fixed.",
                    self._fallback_count,
                    (payload.get("game") or {}).get("id"),
                    payload.get("turn"),
                )
            source = "safe_exception"

        debug: Dict[str, Any] = {}
        t1 = time.perf_counter()
        try:
            if self.move_strategy == "tactics":
                move, debug = tactics.choose_move(payload, preferred=preferred)
                source = f"tactics/{source}"
            elif self.move_strategy == "veto" and ranking:
                # The policy still chooses -- it eats well and grows -- but a
                # short lookahead rejects any choice the opponents can force a
                # loss against. Measured paired against the policy alone, in
                # the same games: +10 points of win rate on two seeds.
                safe = tactics.safe_moves(payload)
                ranked = [m for m in ranking if m in safe] or ranking
                move, sdbg = search.veto(
                    payload, ranked, time_budget_ms=self.search_budget_ms
                )
                if move is None:
                    move = next((m for m in ranked if m in safe), None) or choose_safe_move(
                        payload, preferred=preferred
                    )
                    source = f"veto_fallback/{source}"
                else:
                    source = f"veto/{source}"
                debug = {"model_ranking": ranking, "safe_moves": safe, "search": sdbg}
            elif self.move_strategy == "model" and ranking:
                # Policy decides, restricted to moves that are not suicide and do
                # not lose a head-to-head. Use when the net outplays the search.
                safe = tactics.safe_moves(payload)
                move = next((m for m in ranking if m in safe), None) or choose_safe_move(
                    payload, preferred=preferred
                )
                source = f"model_safe/{source}"
                debug = {"model_ranking": ranking, "safe_moves": safe}
            else:
                move = choose_safe_move(payload, preferred=preferred)
                if self._terminal_state and source == "safe":
                    source = "terminal"
        except Exception:
            self._fallback_count += 1
            logger.exception("Move selection failed; using safe JSON move")
            move = choose_safe_move(payload, preferred=preferred)
            source = "safe_exception"
        t_strategy = (time.perf_counter() - t1) * 1000.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._last_decision = {
            "move": move,
            "preferred": preferred,
            "source": source,
            "strategy": self.move_strategy,
            "legal": legal_moves(payload),
            "fallback_count": self._fallback_count,
            "ms": round(elapsed_ms, 1),
            # Stage breakdown, added to diagnose a latency regression seen only
            # under Railway's container (12.8 ms locally vs 450-1000 ms served)
            # that a local CPU-quota simulation could not reproduce. Remove once
            # the cause is confirmed and fixed.
            "stages_ms": {
                "ensure_game": round(t_ensure, 1),
                "model_move": round(t_model, 1),
                "strategy": round(t_strategy, 1),
            },
            "torch_threads": torch.get_num_threads(),
            "game_id": (payload.get("game") or {}).get("id"),
            "turn": payload.get("turn"),
            **({"tactics": debug} if debug else {}),
        }
        if preferred is not None and move != preferred:
            logger.info(
                "Safe filter overrode model=%s -> %s legal=%s turn=%s",
                preferred,
                move,
                self._last_decision["legal"],
                payload.get("turn"),
            )
        return move

    def _model_move(self, payload: Mapping[str, Any]) -> Optional[str]:
        state, your_pid = request_to_state(
            payload,
            pid_by_snake_id=self._pid_by_snake_id,
            ghosts=self._ghosts,
        )
        self._your_pid = your_pid
        self._update_ghosts(state)
        width, height = _board_dims(payload)
        self._select_env(self._force_players, width, height)

        if your_pid >= len(state.snakes_alive) or not state.snakes_alive[your_pid]:
            # We are already eliminated. The engine still asks for a move on the
            # final turn; answering from the heuristic is correct and is not a
            # failure, so mark it so the health signal does not cry wolf.
            self._terminal_state = True
            return None

        self._env.set_state(state)
        if self._env.is_terminal():
            logger.warning(
                "Env terminal after set_state alive=%s; skipping model",
                state.snakes_alive,
            )
            return None

        obs, _, _ = self._env.get_obs()
        players_here = list(self._env.players_at_turn())
        if your_pid not in players_here:
            return None

        row_idx = players_here.index(your_pid)
        sl = obs[row_idx : row_idx + 1]
        self._model_ranking = self._rank_moves(sl)
        action = self._select_action(sl, your_pid)
        la = list(self._env.available_actions(your_pid))
        if action not in la:
            if not la:
                return None
            action = int(random.choice(la))
        return action_index_to_move(action)

    def _rank_moves(self, obs_slice: np.ndarray) -> List[str]:
        """All four moves ordered by model score (best first)."""
        with torch.no_grad():
            if isinstance(self.model, PPOPolicy):
                scores = self.model.actor_logits(obs_slice).detach().cpu().numpy()[0]
            else:
                scores = self.model(obs_slice).detach().cpu().numpy()[0]
        order = np.argsort(-scores)
        return [ACTION_NAMES[int(a)] for a in order]

    def _select_action(self, obs_slice: np.ndarray, pid: int) -> int:
        la = list(self._env.available_actions(pid))
        if not la:
            return 0

        if isinstance(self.model, (DQN, RainbowDQN)):
            with torch.no_grad():
                q = self.model(obs_slice).detach().cpu().numpy()[0]
            if self.survival_filter:
                return select_survival_action(
                    self._env,
                    pid,
                    q,
                    legal=la,
                    hunger_health=self.hunger_health,
                    strategy=self.combat_strategy,
                )
            return masked_argmax(q, la)

        if isinstance(self.model, PPOPolicy):
            with torch.no_grad():
                logits = self.model.actor_logits(obs_slice)[0].detach().cpu().numpy()
            if self.survival_filter:
                return select_survival_action(
                    self._env,
                    pid,
                    logits,
                    legal=la,
                    hunger_health=self.hunger_health,
                    strategy=self.combat_strategy,
                )
            mask = np.full(logits.shape, -1e9, dtype=np.float32)
            for a in la:
                mask[a] = logits[a]
            return int(mask.argmax())

        with torch.no_grad():
            logits = self.model(obs_slice).detach().cpu().numpy()[0]
        if self.survival_filter:
            return select_survival_action(
                self._env,
                pid,
                logits,
                legal=la,
                hunger_health=self.hunger_health,
                strategy=self.combat_strategy,
            )
        return masked_argmax(logits, la)

    def last_decision(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._last_decision)

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
        "best_checkpoint/ppo_league_best.pt",
    )
    root = Path(__file__).resolve().parents[4]
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    return _resolve_checkpoint_path(p)
