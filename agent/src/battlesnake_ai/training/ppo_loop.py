"""PPO training loop for multi-snake hisss."""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.training.action_selection import masked_argmax, stochastic_joint
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

    def log_evaluation(
        self,
        episode: int,
        *,
        win_rate: float,
        avg_steps: float,
        avg_return: float,
    ) -> None:
        self.logger.info(
            "Evaluation at Episode %s | win_rate=%.1f%% | avg_steps=%.1f | avg_return=%.4f",
            episode,
            win_rate * 100.0,
            avg_steps,
            avg_return,
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
        lr_end: float = 0.0,
        rollout_steps: int = 2048,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        entropy_coef_end: float = 0.001,
        max_grad_norm: float = 0.5,
        survival_shaping: bool = False,
        living_bonus: float = 0.005,
        length_penalty: float = 0.01,
        proximity_penalty: float = 0.005,
        survival_strategy: str = "aggressive",
        device: Optional[torch.device] = None,
        gui: Optional[Any] = None,
        gui_every: int = 1,
        freeze_encoder: bool = False,
        eval_every: int = 0,
        eval_episodes: int = 10,
        rainbow_opponent: Optional[nn.Module] = None,
    ):
        self.env = env
        self.policy = policy
        self.metrics = metrics
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.lr_end = lr_end
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.entropy_coef_end = entropy_coef_end
        self.current_entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.survival_shaping = survival_shaping
        self.living_bonus = living_bonus
        self.length_penalty = length_penalty
        self.proximity_penalty = proximity_penalty
        self.survival_strategy = survival_strategy
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gui = gui
        self.gui_every = gui_every
        # Rainbow DQN opponent model (frozen) used in place of self-play.
        self.rainbow_opponent = rainbow_opponent
        if self.rainbow_opponent is not None:
            self.rainbow_opponent.to(self.device)
            self.rainbow_opponent.eval()

        self.policy.to(self.device)
        if freeze_encoder:
            for p in self.policy.backbone.parameters():
                p.requires_grad = False
        self.optimizer = torch.optim.Adam(
            [p for p in self.policy.parameters() if p.requires_grad], lr=lr
        )
        self.total_env_steps = 0
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes

    def _rainbow_action(self, obs_slice: np.ndarray, pid: int) -> int:
        """Select an action for an opponent snake using the frozen Rainbow DQN."""
        import random
        la = list(self.env.available_actions(pid))
        if not la:
            return 0
        with torch.no_grad():
            q = self.rainbow_opponent(obs_slice).detach().cpu().numpy()[0]
        return masked_argmax(q, la)

    def _select_actions(self, obs: np.ndarray) -> Tuple[Tuple[int, ...], List[float], List[float]]:
        """Select actions for all players.

        Seat 0 always uses the PPO policy being trained.  All other seats use
        the Rainbow DQN opponent when one is provided, otherwise fall back to
        the PPO self-play policy.
        """
        import random
        players = list(self.env.players_at_turn())
        actions: List[int] = []
        log_probs: List[float] = []
        values: List[float] = []

        with torch.no_grad():
            for row_idx, pid in enumerate(players):
                sl = obs[row_idx : row_idx + 1]
                la = self.env.available_actions(pid)
                if self.rainbow_opponent is not None and pid != 0:
                    # Opponent controlled by the frozen Rainbow DQN
                    a = self._rainbow_action(sl, pid)
                    actions.append(a)
                    # Dummy log_prob / value for non-policy snakes (not used in update)
                    log_probs.append(0.0)
                    values.append(0.0)
                else:
                    logits = self.policy.actor_logits(sl)[0]
                    mask = torch.full_like(logits, -1e9)
                    for a in la:
                        mask[a] = 0.0
                    dist = torch.distributions.Categorical(logits=logits + mask)
                    sampled = int(dist.sample().item())
                    actions.append(sampled)
                    log_probs.append(float(dist.log_prob(torch.tensor(sampled, device=logits.device)).item()))
                    values.append(float(self.policy.value(sl).item()))
        return tuple(actions), log_probs, values

    def _min_head_dist(self, st: Any, pid: int, only_ge: bool = False, only_lt: bool = False) -> float:
        """Helper to find distance to nearest opponent head."""
        if not st.snakes_alive[pid]:
            return float("inf")
        my_head = st.snake_pos.get(pid)
        if not my_head or len(my_head) == 0:
            return float("inf")
        hx, hy = my_head[0]
        my_len = int(st.snake_len[pid])

        min_d = float("inf")
        for oid, alive in enumerate(st.snakes_alive):
            if not alive or oid == pid:
                continue
            opp_len = int(st.snake_len[oid])
            if only_ge and opp_len < my_len:
                continue
            if only_lt and opp_len >= my_len:
                continue

            opp_head = st.snake_pos.get(oid)
            if opp_head and len(opp_head) > 0:
                ohx, ohy = opp_head[0]
                d = abs(hx - ohx) + abs(hy - ohy)
                if d < min_d:
                    min_d = float(d)
        return min_d

    def _reshape_reward(
        self,
        pid: int,
        base_reward: float,
        *,
        st_before: Any,
        st_after: Any,
        died: bool,
    ) -> float:
        """Optional reward shaping for grow-then-hunt or pure survival."""
        if not self.survival_shaping:
            return float(base_reward)
        r = float(base_reward)
        if died:
            return r

        aggressive = self.survival_strategy not in {"defensive", "survive", "survival"}
        r += self.living_bonus
        
        # Determine early vs endgame (based on number of snakes alive)
        num_alive = sum(1 for alive in st_before.snakes_alive if alive)
        endgame = num_alive <= 2

        # 1. Starvation Urgency
        health_before = int(st_before.snake_health[pid])
        len_before = int(st_before.snake_len[pid])
        len_after = int(st_after.snake_len[pid]) if st_after.snakes_alive[pid] else len_before
        
        ate_food = len_after > len_before
        if ate_food:
            if aggressive:
                # Base growth reward
                growth_reward = self.length_penalty 
                # Urgency multiplier: if health was very low, increase the reward
                if health_before < 30:
                    growth_reward *= 3.0
                elif health_before > 80:
                    growth_reward = 0.0   # No reward for being greedy when full
                r += growth_reward
            else:
                r -= self.length_penalty

        # 2. Board Control (Edge Penalty when threatened)
        # Check if head is on the edge of the board
        head_pos = st_after.snake_pos.get(pid)
        if head_pos and len(head_pos) > 0:
            hx, hy = int(head_pos[0][0]), int(head_pos[0][1])
            w, h = self.env.cfg.w, self.env.cfg.h
            on_edge = (hx == 0 or hx == w - 1 or hy == 0 or hy == h - 1)
            
            if on_edge:
                # Check if threatened by a larger/equal snake
                dist_to_threat = self._min_head_dist(st_after, pid, only_ge=True)
                if dist_to_threat <= 3:
                    r -= 0.1  # Penalize being trapped on the edge near a threat

        # 3. Kill Bonus
        for oid, alive_before in enumerate(st_before.snakes_alive):
            if oid != pid and alive_before and not st_after.snakes_alive[oid]:
                # Opponent died. Check if we were adjacent to them in st_before
                opp_body = st_before.snake_pos.get(oid) or []
                our_body = st_before.snake_pos.get(pid) or []
                if opp_body and our_body:
                    ohx, ohy = int(opp_body[0][0]), int(opp_body[0][1])
                    # Distance from their head to any part of our body
                    min_dist = min(abs(ohx - int(bx)) + abs(ohy - int(by)) for bx, by in our_body)
                    if min_dist <= 2:
                        r += 0.5  # Massive reward for eliminating an opponent nearby
                        break

        # 4. Proximity strategy
        if st_after.snakes_alive[pid]:
            # Avoid closing on equal/longer heads.
            d0 = self._min_head_dist(st_before, pid, only_ge=True)
            d1 = self._min_head_dist(st_after, pid, only_ge=True)
            if np.isfinite(d0) and np.isfinite(d1) and d1 < d0:
                # Heavy penalty if not endgame
                multiplier = 1.0 if endgame else 2.0
                r -= self.proximity_penalty * (d0 - d1) * multiplier

            if aggressive:
                # Reward closing on shorter prey (hunt).
                p0 = self._min_head_dist(st_before, pid, only_lt=True)
                p1 = self._min_head_dist(st_after, pid, only_lt=True)
                if np.isfinite(p0) and np.isfinite(p1) and p1 < p0:
                    # Double hunting reward in endgame
                    multiplier = 2.0 if endgame else 0.0  # In early game, don't hunt
                    r += self.proximity_penalty * (p0 - p1) * multiplier

        return float(np.clip(r, -1.0, 1.0))

    def ppo_update(self, buffer: RolloutBuffer, last_value: float, episode_idx: int) -> Dict[str, float]:
        advantages, returns = buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_arr = _stack_obs([s.obs for s in buffer.steps])
        actions = torch.tensor([s.action for s in buffer.steps], dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(
            [s.log_prob for s in buffer.steps], dtype=torch.float32, device=self.device
        )
        old_values = torch.tensor(
            [s.value for s in buffer.steps], dtype=torch.float32, device=self.device
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
                mb_old_values = old_values[idx]
                mb_returns = returns_t[idx]
                mb_adv = advantages_t[idx]

                self.optimizer.zero_grad(set_to_none=True)
                log_probs, values, entropy = self.policy.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(log_probs - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = mb_old_values + torch.clamp(values - mb_old_values, -self.clip_eps, self.clip_eps)
                v_loss1 = nn.functional.mse_loss(values, mb_returns, reduction="none")
                v_loss2 = nn.functional.mse_loss(v_clipped, mb_returns, reduction="none")
                value_loss = torch.max(v_loss1, v_loss2).mean()

                ent = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.current_entropy_coef * ent
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

        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=self.lr_end / self.lr if self.lr > 0 else 0.0,
            total_iters=max(1, num_episodes)
        )

        while episode_idx < num_episodes:
            progress = min(1.0, episode_idx / max(1, num_episodes))
            self.current_entropy_coef = self.entropy_coef + progress * (self.entropy_coef_end - self.entropy_coef)
            self.env.reset()
            ep_returns = np.zeros(self.env.num_players, dtype=np.float64)
            ep_steps = 0
            done = False

            while len(buffer) < self.rollout_steps:
                obs, _, _ = self.env.get_obs()
                pat = list(self.env.players_at_turn())
                actions, log_probs, values = self._select_actions(obs)
                st_before = self.env.get_state()
                rewards, done, _ = self.env.step(actions)
                st_after = self.env.get_state()
                
                self.total_env_steps += 1
                ep_steps += 1
                ep_returns += rewards

                for row_idx, pid in enumerate(pat):
                    raw_reward = float(rewards[pid])
                    died = not st_after.snakes_alive[pid]
                    shaped_reward = self._reshape_reward(
                        pid, raw_reward, st_before=st_before, st_after=st_after, died=died
                    )
                    buffer.push(
                        RolloutStep(
                            obs=obs[row_idx].copy(),
                            action=actions[row_idx],
                            log_prob=log_probs[row_idx],
                            reward=shaped_reward,
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
            self.lr_scheduler.step()
            self.metrics.log_episode_end(episode_idx, ep_steps, ep_returns)
            if on_episode_end is not None:
                on_episode_end(episode_idx)
            if self.eval_every > 0 and episode_idx % self.eval_every == 0:
                eval_stats = self.evaluate_policy(self.eval_episodes)
                self.metrics.log_evaluation(
                    episode_idx,
                    win_rate=eval_stats["win_rate"],
                    avg_steps=eval_stats["avg_steps"],
                    avg_return=eval_stats["avg_return"],
                )

    def evaluate_policy(self, num_episodes: int = 10) -> Dict[str, float]:
        import hisss
        import random

        was_training = self.policy.training
        self.policy.eval()

        eval_env = hisss.BattleSnakeGame(self.env.cfg)

        wins = 0
        total_steps = 0
        total_returns = 0.0

        for ep in range(num_episodes):
            eval_env.reset()
            done = False
            steps = 0
            last_rewards = None

            # Policy controls seat 0
            seat = 0

            while not done:
                obs, _, _ = eval_env.get_obs()
                pat = list(eval_env.players_at_turn())

                if seat not in pat:
                    # Policy is dead — opponents finish the game
                    actions = []
                    for row_idx, pid in enumerate(pat):
                        sl = obs[row_idx : row_idx + 1]
                        if self.rainbow_opponent is not None:
                            la = list(eval_env.available_actions(pid))
                            with torch.no_grad():
                                q = self.rainbow_opponent(sl).detach().cpu().numpy()[0]
                            actions.append(masked_argmax(q, la))
                        else:
                            la = eval_env.available_actions(pid)
                            actions.append(int(random.choice(la)))
                    joint = tuple(actions)
                    legal = [tuple(x) for x in eval_env.available_joint_actions()]
                    if joint not in legal:
                        joint = tuple(random.choice(legal))
                    last_rewards, done, _ = eval_env.step(joint)
                    steps += 1
                    continue

                actions = []
                for row_idx, pid in enumerate(pat):
                    sl = obs[row_idx : row_idx + 1]
                    if pid == seat:
                        # PPO policy picks greedily
                        with torch.no_grad():
                            logits = self.policy.actor_logits(sl).detach().cpu().numpy()[0]
                        la = eval_env.available_actions(pid)
                        best = la[0]
                        best_v = logits[best]
                        for a in la[1:]:
                            if logits[a] > best_v:
                                best_v = logits[a]
                                best = a
                        actions.append(int(best))
                    else:
                        # Opponent: Rainbow DQN or random
                        if self.rainbow_opponent is not None:
                            la = list(eval_env.available_actions(pid))
                            with torch.no_grad():
                                q = self.rainbow_opponent(sl).detach().cpu().numpy()[0]
                            actions.append(masked_argmax(q, la))
                        else:
                            la = eval_env.available_actions(pid)
                            actions.append(int(random.choice(la)))

                joint = tuple(actions)
                legal = [tuple(x) for x in eval_env.available_joint_actions()]
                if joint not in legal:
                    joint = tuple(random.choice(legal))

                last_rewards, done, _ = eval_env.step(joint)
                steps += 1

            if last_rewards is not None:
                rewards_list = [float(last_rewards[i]) for i in range(len(last_rewards))]
                seat_reward = rewards_list[seat]
                other_rewards = [rewards_list[i] for i in range(len(rewards_list)) if i != seat]
                if len(other_rewards) == 0:
                    wins += 1
                elif seat_reward > max(other_rewards):
                    wins += 1
                total_returns += seat_reward

            total_steps += steps

        win_rate = wins / num_episodes
        avg_steps = total_steps / num_episodes
        avg_return = total_returns / num_episodes

        self.policy.train(was_training)

        return {
            "win_rate": win_rate,
            "avg_steps": avg_steps,
            "avg_return": avg_return,
        }

