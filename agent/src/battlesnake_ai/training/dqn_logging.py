"""
Structured logging for DQN: Bellman/TD diagnostics, rewards, and hyperparameters.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore[misc, assignment]


class DQNMetricsLogger:
    """
    Writes JSONL metric rows, TensorBoard scalars,
    and optional DEBUG lines with numeric TD examples.
    """

    def __init__(
        self,
        logger: logging.Logger,
        log_dir: str,
        tensorboard_writer: Optional["SummaryWriter"] = None,
        jsonl_name: str = "dqn_metrics.jsonl",
    ):
        self.logger = logger
        self.tb = tensorboard_writer
        os.makedirs(log_dir, exist_ok=True)
        self._jsonl_path = os.path.join(log_dir, jsonl_name)
        self._global_step = 0

    def log_training_startup(self, config: Dict[str, Any]) -> None:
        self.logger.info("DQN training configuration: %s", json.dumps(config, indent=2))

    def log_grad_step(
        self,
        *,
        loss: float,
        td_errors_mean: float,
        td_errors_abs_mean: float,
        q_policy_mean: float,
        q_target_max_mean: float,
        gamma: float,
        epsilon: float,
        batch_size: int,
        grad_norm: Optional[float],
        rewards_in_batch_mean: float,
        sample_td_detail: Optional[str] = None,
        log_level_console: str = "info",
    ) -> None:
        self._global_step += 1
        row = {
            "step": self._global_step,
            "ts": datetime.utcnow().isoformat() + "Z",
            "loss": loss,
            "td_errors_mean": td_errors_mean,
            "td_errors_abs_mean": td_errors_abs_mean,
            "q_policy_mean": q_policy_mean,
            "q_target_max_mean": q_target_max_mean,
            "gamma": gamma,
            "epsilon": epsilon,
            "batch_size": batch_size,
            "grad_norm": grad_norm,
            "rewards_in_batch_mean": rewards_in_batch_mean,
        }
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        msg = (
            f"DQN update | loss={loss:.5f} | TD mean={td_errors_mean:.5f} | "
            f"|TD| mean={td_errors_abs_mean:.5f} | Q(s,a) mean={q_policy_mean:.5f} | "
            f"max_a' Q_tgt(s') mean={q_target_max_mean:.5f} | γ={gamma} | ε={epsilon:.4f} | "
            f"batch={batch_size} | E[r_batch]={rewards_in_batch_mean:.5f}"
        )
        if grad_norm is not None:
            msg += f" | ||∇||={grad_norm:.5f}"
        log_fn = self.logger.info if log_level_console == "info" else self.logger.debug
        log_fn(msg)

        if sample_td_detail:
            self.logger.debug("TD detail (one sample): %s", sample_td_detail)

        if self.tb is not None:
            self.tb.add_scalar("DQN/loss", loss, self._global_step)
            self.tb.add_scalar("DQN/td_error_mean", td_errors_mean, self._global_step)
            self.tb.add_scalar("DQN/td_abs_mean", td_errors_abs_mean, self._global_step)
            self.tb.add_scalar("DQN/q_policy_mean", q_policy_mean, self._global_step)
            self.tb.add_scalar("DQN/q_target_max_mean", q_target_max_mean, self._global_step)
            self.tb.add_scalar("DQN/epsilon", epsilon, self._global_step)
            if grad_norm is not None:
                self.tb.add_scalar("DQN/grad_norm", grad_norm, self._global_step)

    def log_env_step_reward(
        self,
        *,
        turn: int,
        rewards: Any,
        done: bool,
        actions: tuple,
    ) -> None:
        self.logger.debug(
            "env step | turn=%s | actions=%s | rewards=%s | done=%s",
            turn,
            actions,
            rewards,
            done,
        )

    def log_episode_end(
        self,
        episode: int,
        *,
        total_steps: int,
        episode_reward_sum: Any,
        epsilon: float,
        buffer_size: int,
    ) -> None:
        self.logger.info(
            "Episode %s finished | env steps=%s | cumulative reward (per snake)=%s | "
            "ε=%.4f | replay size=%s",
            episode,
            total_steps,
            episode_reward_sum,
            epsilon,
            buffer_size,
        )
        if self.tb is not None:
            self.tb.add_scalar("Episode/steps", total_steps, episode)
            for i, r in enumerate(episode_reward_sum):
                self.tb.add_scalar(f"Episode/reward_snake_{i}", float(r), episode)
            self.tb.add_scalar("Episode/epsilon", epsilon, episode)
