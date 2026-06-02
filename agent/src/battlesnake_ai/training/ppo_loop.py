"""PPO training loop for multi-snake hisss."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.training.action_selection import stochastic_joint
from battlesnake_ai.training.rollout_buffer import RolloutBuffer, RolloutStep


class PPOMetricsLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def log_training_startup(self, cfg: dict) -> None:
        self.logger.info("PPO training configuration: %s", cfg)

    def log_episode_end(self, episode: int, total_steps: int, returns: np.ndarray) -> None:
        self.logger.info(
            "Episode %s finished | env steps=%s | cumulative reward (per snake)=%s",
            episode,
            total_steps,
            returns,
        )

    def log_update(self, episode: int, loss: float, policy_loss: float, value_loss: float, entropy: float) -> None:
        self.logger.info(
            "PPO update | ep=%s | loss=%.5f | policy=%.5f | value=%.5f | entropy=%.5f",
            episode,
            loss,
            policy_loss,
            value_loss,
            entropy,
        )


def _stack_obs(obs_list: List[np.ndarray]) -> np.ndarray:
    return np.stack(obs_list, axis=0)


class PPOTrainingLoop:
    def __init__(
        self,
        env: Any,
        policy: PPOPolicy,
        metrics: PPOMetricsLogger,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        rollout_steps: int = 2048,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: Optional[torch.device] = None,
        gui: Optional[Any] = None,
        gui_every: int = 1,
        freeze_encoder: bool = False,
    ):
        self.env = env
        self.policy = policy
        self.metrics = metrics
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gui = gui
        self.gui_every = gui_every

        self.policy.to(self.device)
        if freeze_encoder:
            for p in self.policy.backbone.parameters():
                p.requires_grad = False
        self.optimizer = torch.optim.Adam(
            [p for p in self.policy.parameters() if p.requires_grad], lr=lr
        )
        self.total_env_steps = 0

    def _select_actions(self, obs: np.ndarray) -> Tuple[Tuple[int, ...], List[float], List[float]]:
        players = list(self.env.players_at_turn())

        def logits_fn(slice_obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return self.policy.actor_logits(slice_obs).detach().cpu().numpy()[0]

        actions = stochastic_joint(self.env, obs, logits_fn)
        log_probs: List[float] = []
        values: List[float] = []
        with torch.no_grad():
            for row_idx, pid in enumerate(players):
                sl = obs[row_idx : row_idx + 1]
                la = self.env.available_actions(pid)
                logits = self.policy.actor_logits(sl)[0]
                mask = torch.full_like(logits, -1e9)
                for a in la:
                    mask[a] = 0.0
                dist = torch.distributions.Categorical(logits=logits + mask)
                a = actions[row_idx]
                log_probs.append(float(dist.log_prob(torch.tensor(a, device=logits.device)).item()))
                values.append(float(self.policy.value(sl).item()))
        return tuple(actions), log_probs, values

    def ppo_update(self, buffer: RolloutBuffer, last_value: float, episode_idx: int) -> Dict[str, float]:
        advantages, returns = buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_arr = _stack_obs([s.obs for s in buffer.steps])
        actions = torch.tensor([s.action for s in buffer.steps], dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(
            [s.log_prob for s in buffer.steps], dtype=torch.float32, device=self.device
        )
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        n = len(buffer)
        total_loss = 0.0
        total_pl = 0.0
        total_vl = 0.0
        total_ent = 0.0
        updates = 0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                idx = indices[start:end]
                mb_obs = obs_arr[idx]
                mb_actions = actions[idx]
                mb_old_log = old_log_probs[idx]
                mb_returns = returns_t[idx]
                mb_adv = advantages_t[idx]

                self.optimizer.zero_grad(set_to_none=True)
                log_probs, values, entropy = self.policy.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(log_probs - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)
                ent = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                total_pl += float(policy_loss.item())
                total_vl += float(value_loss.item())
                total_ent += float(ent.item())
                updates += 1

        stats = {
            "loss": total_loss / max(updates, 1),
            "policy_loss": total_pl / max(updates, 1),
            "value_loss": total_vl / max(updates, 1),
            "entropy": total_ent / max(updates, 1),
        }
        self.metrics.log_update(
            episode_idx,
            stats["loss"],
            stats["policy_loss"],
            stats["value_loss"],
            stats["entropy"],
        )
        return stats

    def train(
        self,
        num_episodes: int,
        *,
        on_episode_end: Optional[Callable[[int], None]] = None,
    ) -> None:
        cfg = {
            "algorithm": "ppo",
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "rollout_steps": self.rollout_steps,
            "ppo_epochs": self.ppo_epochs,
            "clip_eps": self.clip_eps,
        }
        self.metrics.log_training_startup(cfg)
        buffer = RolloutBuffer()
        episode_idx = 0

        while episode_idx < num_episodes:
            self.env.reset()
            ep_returns = np.zeros(self.env.num_players, dtype=np.float64)
            ep_steps = 0
            done = False

            while len(buffer) < self.rollout_steps:
                obs, _, _ = self.env.get_obs()
                pat = list(self.env.players_at_turn())
                actions, log_probs, values = self._select_actions(obs)
                rewards, done, _ = self.env.step(actions)
                self.total_env_steps += 1
                ep_steps += 1
                ep_returns += rewards

                for row_idx, pid in enumerate(pat):
                    buffer.push(
                        RolloutStep(
                            obs=obs[row_idx].copy(),
                            action=actions[row_idx],
                            log_prob=log_probs[row_idx],
                            reward=float(rewards[pid]),
                            done=done,
                            value=values[row_idx],
                        )
                    )

                if self.gui is not None and self.gui_every > 0 and self.total_env_steps % self.gui_every == 0:
                    try:
                        self.gui.update_from_env(
                            self.env,
                            hud={"episode": episode_idx + 1, "env_steps": self.total_env_steps},
                        )
                    except Exception:
                        pass

                if done:
                    break

            last_value = 0.0
            if not done:
                next_obs, _, _ = self.env.get_obs()
                with torch.no_grad():
                    last_value = float(self.policy.value(next_obs[0:1]).item())

            if len(buffer) >= self.minibatch_size:
                self.ppo_update(buffer, last_value, episode_idx + 1)
                buffer.clear()

            episode_idx += 1
            self.metrics.log_episode_end(episode_idx, ep_steps, ep_returns)
            if on_episode_end is not None:
                on_episode_end(episode_idx)
