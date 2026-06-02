"""Rainbow DQN training loop (PER, n-step, distributional, dueling, double Q)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from battlesnake_ai.models.rainbow_dqn import RainbowDQN
from battlesnake_ai.training.action_selection import select_joint_actions_epsilon_greedy
from battlesnake_ai.training.dqn_logging import DQNMetricsLogger
from battlesnake_ai.training.prioritized_replay import (
    PrioritizedReplayBuffer,
    PrioritizedTransition,
)
from battlesnake_ai.viz.board_gui import action_tuple_label


def _terminal_next_obs_like(obs: np.ndarray) -> np.ndarray:
    return np.zeros_like(obs)


def _stack_batch_obs(obs_list: List[np.ndarray]) -> np.ndarray:
    return np.stack(obs_list, axis=0)


class RainbowTrainingLoop:
    def __init__(
        self,
        env: Any,
        policy_net: RainbowDQN,
        target_net: RainbowDQN,
        replay: PrioritizedReplayBuffer,
        metrics: DQNMetricsLogger,
        *,
        gamma: float = 0.99,
        n_step: int = 3,
        lr: float = 1e-4,
        batch_size: int = 64,
        train_after: int = 500,
        train_every: int = 1,
        target_update_every: int = 1000,
        max_grad_norm: float = 10.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 50_000,
        console_log_every: int = 10,
        device: Optional[torch.device] = None,
        gui: Optional[Any] = None,
        gui_every: int = 1,
    ):
        self.env = env
        self.policy = policy_net
        self.target = target_net
        self.replay = replay
        self.metrics = metrics
        self.gamma = gamma
        self.n_step = max(1, n_step)
        self.batch_size = batch_size
        self.train_after = train_after
        self.train_every = train_every
        self.target_update_every = target_update_every
        self.max_grad_norm = max_grad_norm
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.console_log_every = max(1, int(console_log_every))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gui = gui
        self.gui_every = gui_every
        self.lr = lr

        self.policy.to(self.device)
        self.target.to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.total_env_steps = 0
        self.optim_steps = 0

    def epsilon_by_step(self) -> float:
        if self.total_env_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        t = self.total_env_steps / max(1, self.epsilon_decay_steps)
        return self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    def beta_by_step(self) -> float:
        if self.total_env_steps >= self.beta_anneal_steps:
            return self.beta_end
        t = self.total_env_steps / max(1, self.beta_anneal_steps)
        return self.beta_start + t * (self.beta_end - self.beta_start)

    def _push_transition(
        self,
        obs_snake: np.ndarray,
        action: int,
        reward: float,
        next_obs_snake: Optional[np.ndarray],
        done: bool,
    ) -> None:
        """Push completed n-step style transition (used after episode step logic)."""
        if done or next_obs_snake is None:
            nxt = _terminal_next_obs_like(obs_snake)
            done_flag = True
            gamma_n = 1.0
        else:
            nxt = next_obs_snake
            done_flag = False
            gamma_n = self.gamma**self.n_step

        ret = float(reward)
        if not done_flag and self.n_step > 1:
            ret = float(reward)

        self.replay.push(
            PrioritizedTransition(
                obs=obs_snake,
                action=action,
                reward=ret,
                next_obs=nxt,
                done=done_flag,
                n_step_return=ret,
                gamma_n=float(gamma_n),
            )
        )

    def train_step(self, epsilon: float, beta: float) -> Optional[Dict[str, float]]:
        if len(self.replay) < self.train_after or len(self.replay) < self.batch_size:
            return None
        if self.total_env_steps % self.train_every != 0:
            return None

        batch, indices, is_weights = self.replay.sample(self.batch_size, beta)
        obs_b = _stack_batch_obs([t.obs for t in batch])
        next_b = _stack_batch_obs([t.next_obs for t in batch])
        a = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device)
        r = torch.tensor([t.n_step_return for t in batch], dtype=torch.float32, device=self.device)
        d = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        gamma_n = torch.tensor([t.gamma_n for t in batch], dtype=torch.float32, device=self.device)
        w = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)

        log_dist = self.policy.forward_dist(obs_b)
        batch_n = log_dist.shape[0]
        log_prob_a = log_dist[torch.arange(batch_n, device=self.device), a]

        with torch.no_grad():
            next_log_dist_policy = self.policy.forward_dist(next_b)
            q_next = (next_log_dist_policy.exp() * self.policy.support).sum(dim=2)
            best_next = q_next.argmax(dim=1)
            next_log_dist_target = self.target.forward_dist(next_b)
            next_log_prob = next_log_dist_target[torch.arange(batch_n, device=self.device), best_next]
            target_log = self.target.project_distribution(next_log_prob, r, d, gamma_n)

        loss_vec = -(target_log.exp() * log_prob_a).sum(dim=1)
        loss = (loss_vec * w).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_steps += 1

        with torch.no_grad():
            td = (target_log.exp() - log_prob_a.exp()).sum(dim=1).detach().cpu().numpy()

        self.replay.update_priorities(indices, td)

        if self.optim_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.policy.state_dict())

        q_mean = float(
            (log_prob_a.exp() * self.policy.support).sum(dim=1).mean().item()
        )
        stats = {
            "loss": float(loss.item()),
            "td_errors_mean": float(td.mean()),
            "td_errors_abs_mean": float(np.abs(td).mean()),
            "q_policy_mean": q_mean,
            "q_target_max_mean": q_mean,
            "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "rewards_in_batch_mean": float(r.mean().item()),
            "sample_td_detail": None,
        }

        console_level = "info" if (self.optim_steps % self.console_log_every == 0) else "debug"
        self.metrics.log_grad_step(
            loss=stats["loss"],
            td_errors_mean=stats["td_errors_mean"],
            td_errors_abs_mean=stats["td_errors_abs_mean"],
            q_policy_mean=stats["q_policy_mean"],
            q_target_max_mean=stats["q_target_max_mean"],
            gamma=self.gamma,
            epsilon=epsilon,
            batch_size=self.batch_size,
            grad_norm=stats["grad_norm"],
            rewards_in_batch_mean=stats["rewards_in_batch_mean"],
            sample_td_detail=None,
            log_level_console=console_level,
        )
        return stats

    def run_episode(self, episode_idx: int) -> Dict[str, Any]:
        self.env.reset()
        ep_returns = np.zeros(self.env.num_players, dtype=np.float64)
        turns = 0
        done = False
        last_actions: Tuple[int, ...] = ()

        while not done:
            obs, _, _ = self.env.get_obs()
            pat = list(self.env.players_at_turn())
            if obs.shape[0] != len(pat):
                raise RuntimeError(f"obs rows ({obs.shape[0]}) != players_at_turn ({pat})")

            eps = self.epsilon_by_step()
            beta = self.beta_by_step()

            actions = select_joint_actions_epsilon_greedy(self.env, self.policy, obs, eps)
            last_actions = tuple(actions)
            st_before = self.env.get_state()
            rewards, done, _ = self.env.step(actions)
            turns += 1
            self.total_env_steps += 1
            ep_returns += rewards

            self.metrics.log_env_step_reward(
                turn=int(st_before.turn),
                rewards=rewards,
                done=done,
                actions=actions,
            )

            next_obs = None
            next_pat: Optional[List[int]] = None
            if not done:
                next_obs, _, _ = self.env.get_obs()
                next_pat = list(self.env.players_at_turn())

            for row_idx, pid in enumerate(pat):
                o = obs[row_idx]
                a = actions[row_idx]
                r = float(rewards[pid])
                if done:
                    self._push_transition(o, a, r, None, True)
                elif next_pat is not None and pid not in next_pat:
                    self._push_transition(o, a, r, None, True)
                else:
                    assert next_obs is not None and next_pat is not None
                    ni = next_pat.index(pid)
                    nxt = next_obs[ni]
                    self._push_transition(o, a, r, nxt, False)

            train_stats = self.train_step(eps, beta)

            if self.gui is not None and self.gui_every > 0 and self.total_env_steps % self.gui_every == 0:
                try:
                    st = self.env.get_state()
                    hud: Dict[str, Any] = {
                        "episode": episode_idx,
                        "turn": int(st.turn),
                        "env_steps": self.total_env_steps,
                        "optim_steps": self.optim_steps,
                        "epsilon": eps,
                        "gamma": self.gamma,
                        "replay_fill": f"{len(self.replay)}/{self.replay.capacity}",
                        "action_joint": actions,
                        "step_rewards": np.asarray(rewards).tolist(),
                        "ep_returns": ep_returns.tolist(),
                        "snakes_alive": list(st.snakes_alive),
                        "snake_health": [int(x) for x in st.snake_health],
                        "done": done,
                        "had_training": train_stats is not None,
                    }
                    if train_stats is not None:
                        hud.update(
                            {
                                "loss": train_stats["loss"],
                                "td_mean": train_stats["td_errors_mean"],
                                "td_abs": train_stats["td_errors_abs_mean"],
                                "q_mean": train_stats["q_policy_mean"],
                                "grad_norm": train_stats["grad_norm"],
                            }
                        )
                    self.gui.update_from_env(self.env, hud=hud)
                except Exception:
                    pass

        self.metrics.log_episode_end(
            episode_idx,
            total_steps=turns,
            episode_reward_sum=ep_returns,
            epsilon=self.epsilon_by_step(),
            buffer_size=len(self.replay),
        )
        if self.gui is not None:
            try:
                al = action_tuple_label(last_actions) if last_actions else "—"
                self.gui.note(f"Episode {episode_idx} done | turns={turns} returns={ep_returns.tolist()}")
            except Exception:
                pass
        return {"turns": turns, "rewards": ep_returns}

    def train(
        self,
        num_episodes: int,
        *,
        on_episode_end: Optional[Callable[[int], None]] = None,
    ) -> None:
        cfg = {
            "algorithm": "rainbow",
            "gamma": self.gamma,
            "n_step": self.n_step,
            "batch_size": self.batch_size,
            "train_after": self.train_after,
            "target_update_every": self.target_update_every,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "replay_capacity": self.replay.capacity,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        }
        self.metrics.log_training_startup(cfg)

        for ep in range(1, num_episodes + 1):
            self.run_episode(ep)
            if on_episode_end is not None:
                on_episode_end(ep)
