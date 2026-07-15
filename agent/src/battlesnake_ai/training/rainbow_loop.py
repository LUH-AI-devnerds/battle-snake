"""Rainbow DQN training loop (PER, n-step, distributional, dueling, double Q)."""

from __future__ import annotations

import random
from collections import deque
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
        eval_every: int = 0,
        eval_episodes: int = 10,
        eval_seed: Optional[int] = None,
        self_eval_every: int = 0,
        self_eval_episodes: int = 20,
        survival_shaping: bool = False,
        living_bonus: float = 0.01,
        length_penalty: float = 0.05,
        proximity_penalty: float = 0.02,
        survival_strategy: str = "aggressive",
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
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.self_eval_every = self_eval_every
        self.self_eval_episodes = self_eval_episodes
        self.survival_shaping = survival_shaping
        self.living_bonus = float(living_bonus)
        self.length_penalty = float(length_penalty)
        self.proximity_penalty = float(proximity_penalty)
        self.survival_strategy = (survival_strategy or "aggressive").strip().lower()
        self.lr = lr

        self.policy.to(self.device)
        self.target.to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.total_env_steps = 0
        self.optim_steps = 0
        self.best_win_rate: float = -1.0
        self.best_episode: int = 0
        self._nstep_buf: Dict[int, deque] = {}
        self._frozen_opponent: Optional[RainbowDQN] = None

    def _min_head_dist(
        self,
        state: Any,
        pid: int,
        *,
        only_ge: bool = False,
        only_lt: bool = False,
    ) -> float:
        body = state.snake_pos.get(pid) or []
        if not body:
            return float("inf")
        hx, hy = int(body[0][0]), int(body[0][1])
        our_len = int(state.snake_len[pid])
        best = float("inf")
        for oid, alive in enumerate(state.snakes_alive):
            if not alive or oid == pid:
                continue
            ob = state.snake_pos.get(oid) or []
            if not ob:
                continue
            elen = int(state.snake_len[oid])
            if only_ge and elen < our_len:
                continue
            if only_lt and elen >= our_len:
                continue
            ox, oy = int(ob[0][0]), int(ob[0][1])
            d = abs(ox - hx) + abs(oy - hy)
            if d < best:
                best = float(d)
        return best

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

        len_before = int(st_before.snake_len[pid])
        len_after = int(st_after.snake_len[pid]) if st_after.snakes_alive[pid] else len_before
        if len_after > len_before:
            # Aggressive: reward growth (size enables hunting). Defensive: penalize.
            r += self.length_penalty if aggressive else -self.length_penalty

        if st_after.snakes_alive[pid]:
            # Avoid closing on equal/longer heads.
            d0 = self._min_head_dist(st_before, pid, only_ge=True)
            d1 = self._min_head_dist(st_after, pid, only_ge=True)
            if np.isfinite(d0) and np.isfinite(d1) and d1 < d0:
                r -= self.proximity_penalty * (d0 - d1)

            if aggressive:
                # Reward closing on shorter prey (hunt).
                p0 = self._min_head_dist(st_before, pid, only_lt=True)
                p1 = self._min_head_dist(st_after, pid, only_lt=True)
                if np.isfinite(p0) and np.isfinite(p1) and p1 < p0:
                    r += self.proximity_penalty * (p0 - p1)

        return float(np.clip(r, -1.0, 1.0))

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

    def _push_aggregated(
        self,
        obs_snake: np.ndarray,
        action: int,
        n_step_return: float,
        next_obs_snake: Optional[np.ndarray],
        done: bool,
        gamma_n: float,
    ) -> None:
        """Push an aggregated n-step transition into the replay buffer."""
        if done or next_obs_snake is None:
            nxt = _terminal_next_obs_like(obs_snake)
            done_flag = True
        else:
            nxt = next_obs_snake
            done_flag = False

        self.replay.push(
            PrioritizedTransition(
                obs=obs_snake,
                action=action,
                reward=n_step_return,
                next_obs=nxt,
                done=done_flag,
                n_step_return=n_step_return,
                gamma_n=float(gamma_n),
            )
        )

    def _nstep_append(self, pid: int, obs: np.ndarray, action: int, reward: float) -> None:
        """Append a single-step experience to this snake's n-step buffer."""
        buf = self._nstep_buf.get(pid)
        if buf is None:
            buf = deque()
            self._nstep_buf[pid] = buf
        buf.append((obs, action, float(reward)))

    def _nstep_emit_ready(self, pid: int) -> None:
        """Emit completed non-terminal n-step transitions once the buffer is full.

        Each entry stores (s_t, a_t, r_{t+1}). When the buffer holds n+1 entries
        (times t-n .. t), the oldest entry (time t-n) has a full n-step return:
        R = sum_{i=0}^{n-1} gamma^i * r_{t-n+1+i}, bootstrapped from V(s_t)
        (the obs of the newest entry).
        """
        buf = self._nstep_buf.get(pid)
        if buf is None:
            return
        n = self.n_step
        while len(buf) > n:
            entries = list(buf)
            n_step_return = 0.0
            for i in range(n):
                n_step_return += (self.gamma ** i) * entries[i][2]
            obs_o = entries[0][0]
            action_o = entries[0][1]
            next_obs_o = entries[-1][0]  # s_{t+n} = newest entry's obs
            gamma_n = self.gamma ** n
            buf.popleft()
            self._push_aggregated(obs_o, action_o, n_step_return, next_obs_o, done=False, gamma_n=gamma_n)

    def _nstep_flush(self, pid: int) -> None:
        """Flush all remaining buffered entries for a snake as terminal transitions.

        Called when the snake dies or the episode ends. Each entry i is emitted
        with a partial k-step return (k = entries from i to end) and done=True.
        """
        buf = self._nstep_buf.get(pid)
        if buf is None:
            return
        entries = list(buf)
        buf.clear()
        total = len(entries)
        for i in range(total):
            k = total - i
            n_step_return = 0.0
            for j in range(k):
                n_step_return += (self.gamma ** j) * entries[i + j][2]
            obs_o = entries[i][0]
            action_o = entries[i][1]
            gamma_n = self.gamma ** k
            self._push_aggregated(obs_o, action_o, n_step_return, None, done=True, gamma_n=gamma_n)
        self._nstep_buf.pop(pid, None)

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
        self._nstep_buf = {}

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
            st_after = self.env.get_state()
            if self.survival_shaping:
                shaped = np.array(rewards, dtype=np.float64, copy=True)
                for pid in range(len(shaped)):
                    died = done or (not st_after.snakes_alive[pid])
                    shaped[pid] = self._reshape_reward(
                        pid,
                        float(shaped[pid]),
                        st_before=st_before,
                        st_after=st_after,
                        died=died,
                    )
                rewards = shaped
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
                snake_died = done or (next_pat is not None and pid not in next_pat)
                if snake_died:
                    # Terminal experience: append then flush as done.
                    self._nstep_append(pid, o, a, r)
                    self._nstep_flush(pid)
                else:
                    assert next_obs is not None and next_pat is not None
                    self._nstep_append(pid, o, a, r)
                    self._nstep_emit_ready(pid)

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

        # Safety flush: any snake still holding buffered n-step entries at episode
        # end (e.g. a snake that never got a final pat entry) is flushed as terminal.
        for pid in list(self._nstep_buf.keys()):
            self._nstep_flush(pid)

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

    def evaluate_policy(
        self,
        num_episodes: int = 10,
        *,
        opponent_policy: Optional[RainbowDQN] = None,
    ) -> Dict[str, float]:
        import hisss

        was_training = self.policy.training
        self.policy.eval()
        if opponent_policy is not None:
            opponent_policy.eval()

        eval_env = hisss.BattleSnakeGame(self.env.cfg)

        wins = 0
        total_steps = 0
        total_returns = 0.0

        for ep in range(num_episodes):
            # Seed each eval episode deterministically when an eval seed is set,
            # so checkpoint rankings are not dominated by eval RNG noise.
            rng_state: Optional[Tuple[Any, Any]] = None
            if self.eval_seed is not None:
                rng_state = (random.getstate(), np.random.get_state())
                random.seed(int(self.eval_seed) + ep)
                np.random.seed(int(self.eval_seed) + ep)

            try:
                eval_env.reset()
                done = False
                steps = 0
                ep_return = 0.0
                last_rewards = None

                # Policy controls seat 0
                seat = 0

                while not done:
                    obs, _, _ = eval_env.get_obs()
                    pat = list(eval_env.players_at_turn())

                    if seat not in pat:
                        # Policy is dead, others play randomly
                        actions = []
                        for pid in pat:
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
                        if pid == seat:
                            sl = obs[row_idx : row_idx + 1]
                            with torch.no_grad():
                                q = self.policy(sl).detach().cpu().numpy()[0]
                            la = eval_env.available_actions(pid)
                            best = la[0]
                            best_v = q[best]
                            for a in la[1:]:
                                if q[a] > best_v:
                                    best_v = q[a]
                                    best = a
                            actions.append(int(best))
                        elif opponent_policy is not None:
                            sl = obs[row_idx : row_idx + 1]
                            with torch.no_grad():
                                q = opponent_policy(sl).detach().cpu().numpy()[0]
                            la = eval_env.available_actions(pid)
                            best = la[0]
                            best_v = q[best]
                            for a in la[1:]:
                                if q[a] > best_v:
                                    best_v = q[a]
                                    best = a
                            actions.append(int(best))
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
            finally:
                if rng_state is not None:
                    random.setstate(rng_state[0])
                    np.random.set_state(rng_state[1])

        win_rate = wins / num_episodes
        avg_steps = total_steps / num_episodes
        avg_return = total_returns / num_episodes

        self.policy.train(was_training)

        return {
            "win_rate": win_rate,
            "avg_steps": avg_steps,
            "avg_return": avg_return,
        }

    def _run_self_eval(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate current policy vs a frozen past-self snapshot (drift check)."""
        import copy

        if self._frozen_opponent is None:
            self._frozen_opponent = RainbowDQN(
                in_channels=self.policy.in_channels,
                num_actions=self.policy.num_actions,
                num_atoms=self.policy.num_atoms,
                v_min=self.policy.v_min,
                v_max=self.policy.v_max,
                feature_dim=self.policy.backbone.feature_dim,
            ).to(self.device)
        self._frozen_opponent.load_state_dict(self.policy.state_dict())
        frozen = copy.deepcopy(self._frozen_opponent)
        try:
            return self.evaluate_policy(num_episodes, opponent_policy=frozen)
        finally:
            del frozen

    def get_training_state(self) -> Dict[str, Any]:
        return {
            "total_env_steps": int(self.total_env_steps),
            "optim_steps": int(self.optim_steps),
            "best_win_rate": float(self.best_win_rate),
            "best_episode": int(self.best_episode),
        }

    def load_training_state(
        self,
        payload: Dict[str, Any],
        *,
        load_optimizer: bool = True,
    ) -> None:
        """Restore training progress so resume continues epsilon/beta annealing."""
        ts = payload.get("training_state", payload)
        self.total_env_steps = int(ts.get("total_env_steps", 0))
        self.optim_steps = int(ts.get("optim_steps", 0))
        self.best_win_rate = float(ts.get("best_win_rate", -1.0))
        self.best_episode = int(ts.get("best_episode", 0))
        if load_optimizer and "optimizer_state_dict" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            except Exception:
                pass

    def train(
        self,
        num_episodes: int,
        *,
        on_episode_end: Optional[Callable[[int], None]] = None,
        on_eval: Optional[Callable[[int, Dict[str, float], bool], None]] = None,
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
            "eval_every": self.eval_every,
            "eval_episodes": self.eval_episodes,
            "survival_shaping": self.survival_shaping,
            "living_bonus": self.living_bonus,
            "length_penalty": self.length_penalty,
            "proximity_penalty": self.proximity_penalty,
            "survival_strategy": self.survival_strategy,
        }
        self.metrics.log_training_startup(cfg)

        for ep in range(1, num_episodes + 1):
            self.run_episode(ep)
            if on_episode_end is not None:
                on_episode_end(ep)
            if self.eval_every > 0 and ep % self.eval_every == 0:
                eval_stats = self.evaluate_policy(self.eval_episodes)
                self.metrics.log_evaluation(
                    ep,
                    win_rate=eval_stats["win_rate"],
                    avg_steps=eval_stats["avg_steps"],
                    avg_return=eval_stats["avg_return"],
                )
                is_best = eval_stats["win_rate"] > self.best_win_rate
                if is_best:
                    self.best_win_rate = eval_stats["win_rate"]
                    self.best_episode = ep
                if on_eval is not None:
                    on_eval(ep, eval_stats, is_best)
            if (
                self.self_eval_every > 0
                and ep % self.self_eval_every == 0
                and ep > 0
            ):
                self_stats = self._run_self_eval(self.self_eval_episodes)
                self.metrics.logger.info(
                    "Evaluation_self at Episode %s | win_rate=%.1f%% | avg_steps=%.1f | avg_return=%.4f",
                    ep,
                    self_stats["win_rate"] * 100.0,
                    self_stats["avg_steps"],
                    self_stats["avg_return"],
                )

