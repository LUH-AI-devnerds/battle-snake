"""PPO training loop for multi-snake hisss.

Design notes (the previous version got these wrong and never learned):

* Only transitions produced by the *learner* are stored.  Opponent seats are
  driven by frozen models / bots and must never enter the buffer.
* Trajectories are kept per seat and GAE runs per seat.  A 4-player game
  interleaves four trajectories in time; bootstrapping across them is garbage.
* The action mask used at sampling time is stored and re-applied in the update,
  so ``ratio = exp(new_logp - old_logp)`` compares the same distribution.
* A seat's trajectory terminates when that snake dies, not only when the whole
  episode ends.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.training.action_selection import masked_argmax
from battlesnake_ai.training.rollout_buffer import MultiSeatRollout, RolloutStep


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

    def log_update(self, episode: int, stats: Dict[str, float]) -> None:
        self.logger.info(
            "PPO update | ep=%s | loss=%.5f | policy=%.5f | value=%.5f | entropy=%.5f "
            "| kl=%.5f | clipfrac=%.3f | n=%s",
            episode,
            stats.get("loss", 0.0),
            stats.get("policy_loss", 0.0),
            stats.get("value_loss", 0.0),
            stats.get("entropy", 0.0),
            stats.get("approx_kl", 0.0),
            stats.get("clip_frac", 0.0),
            int(stats.get("samples", 0)),
        )

    def log_evaluation(self, episode: int, tag: str, stats: Dict[str, float]) -> None:
        self.logger.info(
            "Evaluation[%s] at Episode %s | win_rate=%.1f%% | avg_points=%.3f | "
            "avg_steps=%.1f | avg_len=%.1f",
            tag,
            episode,
            stats["win_rate"] * 100.0,
            stats["avg_points"],
            stats["avg_steps"],
            stats["avg_len"],
        )


# ── Opponent controllers ──────────────────────────────────────────────────────


class Opponent:
    """Wraps anything that can pick an action for a seat."""

    def __init__(self, name: str, fn: Callable[[Any, np.ndarray, int, int], int]) -> None:
        self.name = name
        self._fn = fn

    def act(self, env: Any, obs: np.ndarray, row_idx: int, pid: int) -> int:
        return self._fn(env, obs, row_idx, pid)


def random_opponent() -> Opponent:
    def _act(env: Any, obs: np.ndarray, row_idx: int, pid: int) -> int:
        legal = list(env.available_actions(pid))
        return int(random.choice(legal)) if legal else 0

    return Opponent("random", _act)


def model_opponent(name: str, model: nn.Module, device: torch.device) -> Opponent:
    """Greedy opponent from a Q-network (Rainbow/DQN) or a PPO policy."""
    model.eval()

    def _act(env: Any, obs: np.ndarray, row_idx: int, pid: int) -> int:
        legal = list(env.available_actions(pid))
        if not legal:
            return 0
        sl = obs[row_idx : row_idx + 1]
        with torch.no_grad():
            if isinstance(model, PPOPolicy):
                scores = model.actor_logits(sl).detach().cpu().numpy()[0]
            else:
                scores = model(sl).detach().cpu().numpy()[0]
        return masked_argmax(scores, legal)

    return Opponent(name, _act)


def bot_opponent(bot: Any) -> Opponent:
    def _act(env: Any, obs: np.ndarray, row_idx: int, pid: int) -> int:
        return int(bot.select_action(env, pid))

    return Opponent(f"bot:{getattr(bot, 'name', 'bot')}", _act)


# ── Training loop ─────────────────────────────────────────────────────────────


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
        rollout_steps: int = 4096,
        ppo_epochs: int = 4,
        minibatch_size: int = 256,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        entropy_coef_end: float = 0.001,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.02,
        survival_shaping: bool = False,
        living_bonus: float = 0.005,
        length_penalty: float = 0.01,
        proximity_penalty: float = 0.005,
        survival_strategy: str = "aggressive",
        device: Optional[torch.device] = None,
        gui: Optional[Any] = None,
        gui_every: int = 1,
        freeze_encoder: bool = False,
        freeze_batchnorm: bool = True,
        eval_every: int = 0,
        eval_episodes: int = 50,
        opponents: Optional[Sequence[Opponent]] = None,
        self_play_prob: float = 0.0,
        eval_opponents: Optional[Sequence[Opponent]] = None,
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
        self.target_kl = target_kl
        self.survival_shaping = survival_shaping
        self.living_bonus = living_bonus
        self.length_penalty = length_penalty
        self.proximity_penalty = proximity_penalty
        self.survival_strategy = survival_strategy
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gui = gui
        self.gui_every = gui_every

        # Opponent league. Empty → pure self-play (every seat is the learner).
        self.opponents: List[Opponent] = list(opponents or [])
        self.eval_opponents: List[Opponent] = list(eval_opponents or self.opponents)
        # Probability of running a fully self-play episode when a league exists.
        self.self_play_prob = self_play_prob if self.opponents else 1.0

        self.policy.to(self.device)
        # The backbone is BatchNorm-based. In train mode BN normalises with the
        # statistics of whatever batch it sees, so a rollout (batch of 1) and an
        # update (batch of 512) evaluate *different functions*: measured
        # KL(collect || update) = 0.15 on an untouched policy, which is exactly
        # the bogus ratio PPO then tries to correct. Train-mode rollouts also
        # drift the running stats the served model uses. Keeping the policy in
        # eval mode makes collection, update and deployment identical.
        self.freeze_batchnorm = freeze_batchnorm
        if self.freeze_batchnorm:
            self.policy.eval()
        if freeze_encoder:
            for p in self.policy.backbone.parameters():
                p.requires_grad = False
        self.optimizer = torch.optim.Adam(
            [p for p in self.policy.parameters() if p.requires_grad], lr=lr
        )
        self.total_env_steps = 0
        self.num_updates = 0
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes

    # ── acting ────────────────────────────────────────────────────────────────

    def _legal_mask(self, legal: Sequence[int]) -> np.ndarray:
        mask = np.zeros(self.policy.num_actions, dtype=bool)
        for a in legal:
            mask[int(a)] = True
        return mask

    def _policy_action(
        self, obs_slice: np.ndarray, legal: Sequence[int]
    ) -> Tuple[int, float, float, np.ndarray]:
        """Sample from the masked policy; return action, log_prob, value, mask."""
        mask_np = self._legal_mask(legal)
        with torch.no_grad():
            features = self.policy._features(obs_slice)
            logits = self.policy.actor(features)[0]
            value = float(self.policy.critic(features).squeeze(-1)[0].item())
            mask_t = torch.as_tensor(mask_np, device=logits.device)
            logits = logits.masked_fill(~mask_t, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = float(dist.log_prob(action).item())
        return int(action.item()), log_prob, value, mask_np

    def _greedy_action(self, obs_slice: np.ndarray, legal: Sequence[int]) -> int:
        with torch.no_grad():
            logits = self.policy.actor_logits(obs_slice).detach().cpu().numpy()[0]
        return masked_argmax(logits, list(legal))

    def _assign_seats(self) -> Dict[int, Optional[Opponent]]:
        """Seat 0 always learns; other seats learn (self-play) or get an opponent."""
        seats: Dict[int, Optional[Opponent]] = {0: None}
        n = self.env.num_players
        if not self.opponents or random.random() < self.self_play_prob:
            for pid in range(1, n):
                seats[pid] = None
            return seats
        for pid in range(1, n):
            seats[pid] = random.choice(self.opponents)
        return seats

    # ── reward shaping ────────────────────────────────────────────────────────

    def _min_head_dist(self, st: Any, pid: int, only_ge: bool = False, only_lt: bool = False) -> float:
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
        """Grow-then-hunt shaping on top of the env reward."""
        if not self.survival_shaping:
            return float(base_reward)
        r = float(base_reward)
        if died:
            return r

        aggressive = self.survival_strategy not in {"defensive", "survive", "survival"}
        r += self.living_bonus

        num_alive = sum(1 for alive in st_before.snakes_alive if alive)
        endgame = num_alive <= 2

        health_before = int(st_before.snake_health[pid])
        len_before = int(st_before.snake_len[pid])
        len_after = int(st_after.snake_len[pid]) if st_after.snakes_alive[pid] else len_before

        if len_after > len_before:
            if aggressive:
                growth_reward = self.length_penalty
                if health_before < 30:
                    growth_reward *= 3.0
                elif health_before > 80:
                    growth_reward = 0.0
                r += growth_reward
            else:
                r -= self.length_penalty

        head_pos = st_after.snake_pos.get(pid)
        if head_pos and len(head_pos) > 0:
            hx, hy = int(head_pos[0][0]), int(head_pos[0][1])
            w, h = self.env.cfg.w, self.env.cfg.h
            if hx == 0 or hx == w - 1 or hy == 0 or hy == h - 1:
                if self._min_head_dist(st_after, pid, only_ge=True) <= 3:
                    r -= 0.1

        for oid, alive_before in enumerate(st_before.snakes_alive):
            if oid != pid and alive_before and not st_after.snakes_alive[oid]:
                opp_body = st_before.snake_pos.get(oid) or []
                our_body = st_before.snake_pos.get(pid) or []
                if opp_body and our_body:
                    ohx, ohy = int(opp_body[0][0]), int(opp_body[0][1])
                    min_dist = min(abs(ohx - int(bx)) + abs(ohy - int(by)) for bx, by in our_body)
                    if min_dist <= 2:
                        r += 0.5
                        break

        if st_after.snakes_alive[pid]:
            d0 = self._min_head_dist(st_before, pid, only_ge=True)
            d1 = self._min_head_dist(st_after, pid, only_ge=True)
            if np.isfinite(d0) and np.isfinite(d1) and d1 < d0:
                multiplier = 1.0 if endgame else 2.0
                r -= self.proximity_penalty * (d0 - d1) * multiplier

            if aggressive:
                p0 = self._min_head_dist(st_before, pid, only_lt=True)
                p1 = self._min_head_dist(st_after, pid, only_lt=True)
                if np.isfinite(p0) and np.isfinite(p1) and p1 < p0:
                    multiplier = 2.0 if endgame else 0.0
                    r += self.proximity_penalty * (p0 - p1) * multiplier

        return float(np.clip(r, -1.0, 1.0))

    # ── update ────────────────────────────────────────────────────────────────

    def ppo_update(self, rollout: MultiSeatRollout, last_values: Dict[int, float], episode_idx: int) -> Dict[str, float]:
        batch = rollout.build(last_values, self.gamma, self.gae_lambda)
        n = len(batch)
        if n < 2:
            return {}
        if self.freeze_batchnorm:
            self.policy.eval()  # same normalisation as when the data was collected
        else:
            self.policy.train()

        adv = batch.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_arr = batch.obs
        actions = torch.as_tensor(batch.actions, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, device=self.device)
        old_values = torch.as_tensor(batch.values, device=self.device)
        returns_t = torch.as_tensor(batch.returns, device=self.device)
        advantages_t = torch.as_tensor(adv, device=self.device)
        masks_t = torch.as_tensor(batch.legal_masks, device=self.device)

        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                  "approx_kl": 0.0, "clip_frac": 0.0}
        updates = 0
        stop = False

        for _ in range(self.ppo_epochs):
            if stop:
                break
            indices = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                idx = indices[start : start + self.minibatch_size]
                if len(idx) < 2:
                    continue
                mb_obs = obs_arr[idx]
                mb_actions = actions[idx]
                mb_old_log = old_log_probs[idx]
                mb_old_values = old_values[idx]
                mb_returns = returns_t[idx]
                mb_adv = advantages_t[idx]
                mb_masks = masks_t[idx]

                log_probs, values, entropy = self.policy.evaluate_actions(
                    mb_obs, mb_actions, legal_mask=mb_masks
                )
                log_ratio = log_probs - mb_old_log
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = mb_old_values + torch.clamp(
                    values - mb_old_values, -self.clip_eps, self.clip_eps
                )
                v_loss1 = nn.functional.mse_loss(values, mb_returns, reduction="none")
                v_loss2 = nn.functional.mse_loss(v_clipped, mb_returns, reduction="none")
                value_loss = torch.max(v_loss1, v_loss2).mean()

                ent = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.current_entropy_coef * ent

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                    clip_frac = float(
                        ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                    )

                totals["loss"] += float(loss.item())
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(ent.item())
                totals["approx_kl"] += approx_kl
                totals["clip_frac"] += clip_frac
                updates += 1

                # Trust-region guard: stop early once the policy has moved far enough.
                if self.target_kl > 0 and approx_kl > 1.5 * self.target_kl:
                    stop = True
                    break

        stats = {k: v / max(updates, 1) for k, v in totals.items()}
        stats["samples"] = float(n)
        self.num_updates += 1
        self.metrics.log_update(episode_idx, stats)
        return stats

    # ── rollout / train ───────────────────────────────────────────────────────

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
            "minibatch_size": self.minibatch_size,
            "clip_eps": self.clip_eps,
            "target_kl": self.target_kl,
            "opponents": [o.name for o in self.opponents],
            "self_play_prob": self.self_play_prob,
        }
        self.metrics.log_training_startup(cfg)

        rollout = MultiSeatRollout(num_actions=self.policy.num_actions)
        seat_map = self._assign_seats()
        episode_idx = 0
        ep_returns = np.zeros(self.env.num_players, dtype=np.float64)
        ep_steps = 0
        self.env.reset()

        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=(self.lr_end / self.lr) if self.lr > 0 else 0.0,
            total_iters=max(1, num_episodes),
        )

        while episode_idx < num_episodes:
            progress = min(1.0, episode_idx / max(1, num_episodes))
            self.current_entropy_coef = self.entropy_coef + progress * (
                self.entropy_coef_end - self.entropy_coef
            )

            obs, _, _ = self.env.get_obs()
            pat = list(self.env.players_at_turn())
            st_before = self.env.get_state()

            actions: List[int] = []
            pending: Dict[int, Dict[str, Any]] = {}
            for row_idx, pid in enumerate(pat):
                legal = list(self.env.available_actions(pid))
                sl = obs[row_idx : row_idx + 1]
                opponent = seat_map.get(pid)
                if opponent is None:
                    a, logp, value, mask = self._policy_action(sl, legal)
                    pending[pid] = {
                        "obs": obs[row_idx].copy(),
                        "action": a,
                        "log_prob": logp,
                        "value": value,
                        "mask": mask,
                    }
                else:
                    a = opponent.act(self.env, obs, row_idx, pid)
                    if legal and a not in legal:
                        a = int(random.choice(legal))
                actions.append(int(a))

            rewards, done, _ = self.env.step(tuple(actions))
            st_after = self.env.get_state()
            self.total_env_steps += 1
            ep_steps += 1
            ep_returns += rewards

            for pid, rec in pending.items():
                died = not st_after.snakes_alive[pid]
                shaped = self._reshape_reward(
                    pid, float(rewards[pid]), st_before=st_before, st_after=st_after, died=died
                )
                rollout.push(
                    pid,
                    RolloutStep(
                        obs=rec["obs"],
                        action=rec["action"],
                        log_prob=rec["log_prob"],
                        reward=shaped,
                        done=bool(died or done),
                        value=rec["value"],
                        legal_mask=rec["mask"],
                    ),
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
                episode_idx += 1
                self.lr_scheduler.step()
                self.metrics.log_episode_end(episode_idx, ep_steps, ep_returns)
                if on_episode_end is not None:
                    on_episode_end(episode_idx)
                if self.eval_every > 0 and episode_idx % self.eval_every == 0:
                    stats = self.evaluate_policy(self.eval_episodes)
                    self.metrics.log_evaluation(episode_idx, "league", stats)
                self.env.reset()
                seat_map = self._assign_seats()
                ep_returns = np.zeros(self.env.num_players, dtype=np.float64)
                ep_steps = 0

            if len(rollout) >= self.rollout_steps:
                last_values = self._bootstrap_values(seat_map) if not done else {}
                self.ppo_update(rollout, last_values, episode_idx + 1)
                rollout.clear()

    def _bootstrap_values(self, seat_map: Dict[int, Optional[Opponent]]) -> Dict[int, float]:
        """V(s) for learner seats still alive when the rollout is truncated."""
        out: Dict[int, float] = {}
        try:
            obs, _, _ = self.env.get_obs()
            pat = list(self.env.players_at_turn())
        except Exception:
            return out
        with torch.no_grad():
            for row_idx, pid in enumerate(pat):
                if seat_map.get(pid) is None:
                    out[pid] = float(self.policy.value(obs[row_idx : row_idx + 1])[0].item())
        return out

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate_policy(
        self,
        num_episodes: int = 50,
        opponents: Optional[Sequence[Opponent]] = None,
    ) -> Dict[str, float]:
        """Greedy policy in seat 0 against sampled opponents; Blackout-style scoring."""
        import hisss

        pool = list(opponents or self.eval_opponents) or [random_opponent()]
        was_training = self.policy.training and not self.freeze_batchnorm
        self.policy.eval()
        eval_env = hisss.BattleSnakeGame(self.env.cfg)

        wins = 0
        points = 0.0
        total_steps = 0
        total_len = 0

        for ep in range(num_episodes):
            eval_env.reset()
            seat_opponents = {pid: pool[(ep + pid) % len(pool)] for pid in range(1, eval_env.num_players)}
            done = False
            while not done:
                obs, _, _ = eval_env.get_obs()
                pat = list(eval_env.players_at_turn())
                acts: List[int] = []
                for row_idx, pid in enumerate(pat):
                    legal = list(eval_env.available_actions(pid))
                    if pid == 0:
                        a = self._greedy_action(obs[row_idx : row_idx + 1], legal)
                    else:
                        a = seat_opponents[pid].act(eval_env, obs, row_idx, pid)
                    if legal and a not in legal:
                        a = int(random.choice(legal))
                    acts.append(int(a))
                joint = tuple(acts)
                legal_joint = [tuple(x) for x in eval_env.available_joint_actions()]
                if joint not in legal_joint:
                    joint = tuple(random.choice(legal_joint))
                _, done, _ = eval_env.step(joint)

            st = eval_env.get_state()
            ranked = []
            for pid in range(eval_env.num_players):
                if st.snakes_alive[pid]:
                    surv = st.turn + 1
                elif pid in getattr(st, "elimination_events", {}) or {}:
                    surv = st.elimination_events[pid].turn
                else:
                    surv = st.turn
                ranked.append((surv, int(st.snake_len[pid]), pid))
            ranked.sort(reverse=True)
            rank = [r[2] for r in ranked].index(0)
            points += [2.0, 1.0, 0.0, 0.0][min(rank, 3)]
            if rank == 0:
                wins += 1
            total_steps += ranked[[r[2] for r in ranked].index(0)][0]
            total_len += int(st.snake_len[0])

        self.policy.train(was_training)
        return {
            "win_rate": wins / max(num_episodes, 1),
            "avg_points": points / max(num_episodes, 1),
            "avg_steps": total_steps / max(num_episodes, 1),
            "avg_len": total_len / max(num_episodes, 1),
        }
