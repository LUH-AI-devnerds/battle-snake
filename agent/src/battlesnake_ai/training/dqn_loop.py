"""
DQN training loop (shared Q-network, independent transitions per snake).

Supports any hisss game with simultaneous moves: one row in ``get_obs()`` per
``players_at_turn()`` id. Joint actions are ε-greedy per snake, corrected to a
legal joint move when the greedy tuple is not allowed.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from battlesnake_ai.models.dqn import DQN
from battlesnake_ai.training.dqn_logging import DQNMetricsLogger
from battlesnake_ai.training.replay_buffer import ReplayBuffer, Transition
from battlesnake_ai.viz.board_gui import action_tuple_label


def _terminal_next_obs_like(obs: np.ndarray) -> np.ndarray:
    return np.zeros_like(obs)


def _joint_legal_set(env: Any) -> Set[Tuple[int, ...]]:
    return set(tuple(x) for x in env.available_joint_actions())


def select_joint_actions_epsilon_greedy(
    env: Any,
    policy: DQN,
    obs: np.ndarray,
    epsilon: float,
) -> Tuple[int, ...]:
    """
    Independent ε-greedy per alive snake on masked Q-values; if the tuple is not jointly legal,
    sample a uniform random legal joint action.

    ``obs`` axis 0 matches ``env.players_at_turn()`` order (same as joint moves).
    """
    joint_set = _joint_legal_set(env)
    players_here = list(env.players_at_turn())

    if random.random() < epsilon:
        return tuple(random.choice(env.available_joint_actions()))

    def masked_argmax(q: np.ndarray, legal: List[int]) -> int:
        best = legal[0]
        best_v = q[best]
        for a in legal[1:]:
            if q[a] > best_v:
                best_v = q[a]
                best = a
        return int(best)

    greedy: List[int] = []
    with torch.no_grad():
        for row_idx, pid in enumerate(players_here):
            q = policy(obs[row_idx : row_idx + 1]).detach().cpu().numpy()[0]
            la = env.available_actions(pid)
            greedy.append(masked_argmax(q, la))
    tup = tuple(greedy)
    if tup in joint_set:
        return tup
    return tuple(random.choice(env.available_joint_actions()))


def _stack_batch_obs(obs_list: List[np.ndarray]) -> np.ndarray:
    return np.stack(obs_list, axis=0)


class DQNTrainingLoop:
    def __init__(
        self,
        env: hisss.BattleSnakeGame,
        policy_net: DQN,
        target_net: DQN,
        replay: ReplayBuffer,
        metrics: DQNMetricsLogger,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        train_after: int = 500,
        train_every: int = 1,
        target_update_every: int = 1000,
        max_grad_norm: float = 10.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
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
        self.batch_size = batch_size
        self.train_after = train_after
        self.train_every = train_every
        self.target_update_every = target_update_every
        self.max_grad_norm = max_grad_norm
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
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

    def _push_transition(
        self,
        obs_snake: np.ndarray,
        action: int,
        reward: float,
        next_obs_snake: Optional[np.ndarray],
        done: bool,
    ) -> None:
        if done or next_obs_snake is None:
            nxt = _terminal_next_obs_like(obs_snake)
            done_flag = True
        else:
            nxt = next_obs_snake
            done_flag = False
        self.replay.push(Transition(obs=obs_snake, action=action, reward=float(reward), next_obs=nxt, done=done_flag))

    def train_step(self, epsilon: float) -> Optional[Dict[str, float]]:
        if len(self.replay) < self.train_after or len(self.replay) < self.batch_size:
            return None
        if self.total_env_steps % self.train_every != 0:
            return None

        batch = self.replay.sample(self.batch_size)
        obs_b = _stack_batch_obs([t.obs for t in batch])
        next_b = _stack_batch_obs([t.next_obs for t in batch])
        a = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device)
        r = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        d = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)

        q_all = self.policy(obs_b)
        q_sa = q_all.gather(1, a.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            q_next_policy = self.policy(next_b)
            q_next_target = self.target(next_b)
            best_next = q_next_policy.argmax(dim=1, keepdim=True)
            max_next = q_next_target.gather(1, best_next).squeeze(1)
            y = r + (1.0 - d) * self.gamma * max_next

        loss = nn.functional.mse_loss(q_sa, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_steps += 1

        with torch.no_grad():
            td = (q_sa - y).detach().cpu().numpy()

        sample_detail = None
        if len(td) > 0:
            i = int(np.argmax(np.abs(td)))
            sample_detail = (
                f"sample[{i}]: r={batch[i].reward:.4f}, done={batch[i].done}, "
                f"Q(s,a)={float(q_sa[i].item()):.4f}, y={float(y[i].item()):.4f}, TD={float(td[i]):.4f}"
            )

        if self.optim_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.policy.state_dict())

        stats = {
            "loss": float(loss.item()),
            "td_errors_mean": float(td.mean()),
            "td_errors_abs_mean": float(np.abs(td).mean()),
            "q_policy_mean": float(q_sa.detach().mean().item()),
            "q_target_max_mean": float(max_next.detach().mean().item()),
            "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "rewards_in_batch_mean": float(r.mean().item()),
            "sample_td_detail": sample_detail,
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
            sample_td_detail=sample_detail,
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

            train_stats = self.train_step(eps)

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
                                "q_tgt_max": train_stats["q_target_max_mean"],
                                "grad_norm": train_stats["grad_norm"],
                                "r_batch_mean": train_stats["rewards_in_batch_mean"],
                                "sample_td_detail": train_stats.get("sample_td_detail"),
                            }
                        )
                    self.gui.update_from_env(self.env, hud=hud)
                    if train_stats is not None and self.optim_steps % self.console_log_every == 0:
                        self.gui.note(
                            f"optim {self.optim_steps} loss={train_stats['loss']:.4f} "
                            f"|TD|={train_stats['td_errors_abs_mean']:.4f} ε={eps:.3f}"
                        )
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
                eps_fin = self.epsilon_by_step()
                al = action_tuple_label(last_actions) if last_actions else "—"
                self.gui.note(
                    f"Episode {episode_idx} done | turns={turns} "
                    f"returns={ep_returns.tolist()} ε={eps_fin:.4f} "
                    f"a_last={al}"
                )
            except Exception:
                pass
        return {"turns": turns, "rewards": ep_returns}

    def train(self, num_episodes: int) -> None:
        cfg = {
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "train_after": self.train_after,
            "train_every": self.train_every,
            "target_update_every": self.target_update_every,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "replay_capacity": self.replay.capacity,
        }
        self.metrics.log_training_startup(cfg)

        for ep in range(1, num_episodes + 1):
            self.run_episode(ep)
