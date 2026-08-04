"""Behaviour-cloning warm start for the PPO policy.

Rationale: the hand-written bots in ``training.heuristic_opponents`` beat every
checkpoint in this repo by a wide margin (aggressive_hunter ~75% win rate vs
~5% for the best Rainbow net).  Starting PPO from a random policy means
spending millions of steps rediscovering flood-fill survival.  Cloning the
teacher first gets the network to bot level in minutes, and PPO fine-tuning
then only has to find the improvements the teacher misses.

The teacher labels every state, including states reached by exploration moves
(DAgger-style), so the policy learns to recover from its own mistakes.

Example:
  python agent/scripts/pretrain_bc.py --iterations 40 --steps-per-iter 4096 \
      --teacher aggressive_hunter --out logs/checkpoints/ppo_bc.pt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import torch
import torch.nn as nn

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.training.checkpoint import load_checkpoint, save_checkpoint
from battlesnake_ai.training.heuristic_opponents import get_bot_by_name
from battlesnake_ai.training.logger import setup_logger
from battlesnake_ai.training.ppo_loop import PPOMetricsLogger, PPOTrainingLoop, bot_opponent

BENCHMARK_LEAGUE = ["aggressive_hunter", "cautious", "flood_fill"]


def collect(env, teacher, policy, device, steps: int, epsilon: float, gamma: float):
    """Roll out episodes; label every seat with the teacher's action.

    Actions actually played are the teacher's, except with probability
    ``epsilon`` where a random legal action is played (state-space coverage) or
    the current policy acts (self-induced states) — the label stays the
    teacher's choice for that state.
    """
    # Roll out with the deployment-time normalisation: a train-mode forward pass
    # on a single observation both misnormalises the output and drifts the
    # BatchNorm running statistics that inference relies on.
    if policy is not None:
        policy.eval()

    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    mask_buf: list[np.ndarray] = []
    ret_buf: list[float] = []

    ep_records: list[list[tuple[int, int]]] = []  # (index into buffers, seat)
    ep_rewards: list[np.ndarray] = []

    env.reset()
    current: list[tuple[int, int]] = []
    ep_reward = np.zeros(env.num_players, dtype=np.float64)

    while len(obs_buf) < steps:
        obs, _, _ = env.get_obs()
        pat = list(env.players_at_turn())
        actions: list[int] = []
        for row_idx, pid in enumerate(pat):
            legal = list(env.available_actions(pid))
            if not legal:
                actions.append(0)
                continue
            label = int(teacher.select_action(env, pid))
            mask = np.zeros(4, dtype=bool)
            for a in legal:
                mask[a] = True
            obs_buf.append(obs[row_idx].copy())
            act_buf.append(label)
            mask_buf.append(mask)
            ret_buf.append(0.0)
            current.append((len(obs_buf) - 1, pid))

            if random.random() < epsilon:
                roll = random.random()
                if roll < 0.5 or policy is None:
                    played = int(random.choice(legal))
                else:
                    with torch.no_grad():
                        logits = policy.actor_logits(obs[row_idx : row_idx + 1])[0].cpu().numpy()
                    played = max(legal, key=lambda a: logits[a])
            else:
                played = label
            actions.append(int(played))

        rewards, done, _ = env.step(tuple(actions))
        ep_reward += rewards

        if done:
            ep_records.append(current)
            ep_rewards.append(ep_reward.copy())
            current = []
            ep_reward = np.zeros(env.num_players, dtype=np.float64)
            env.reset()

    # Monte-Carlo value targets: terminal reward discounted back per seat.
    for records, final in zip(ep_records, ep_rewards):
        per_seat: dict[int, list[int]] = {}
        for idx, pid in records:
            per_seat.setdefault(pid, []).append(idx)
        for pid, idxs in per_seat.items():
            g = float(final[pid])
            for step_back, idx in enumerate(reversed(idxs)):
                ret_buf[idx] = g * (gamma ** step_back)

    return (
        np.stack(obs_buf, axis=0),
        np.asarray(act_buf, dtype=np.int64),
        np.stack(mask_buf, axis=0),
        np.asarray(ret_buf, dtype=np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Behaviour-clone a heuristic teacher into PPOPolicy")
    ap.add_argument("--mode", type=str, default="restricted_standard")
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--teacher", type=str, default="aggressive_hunter")
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--steps-per-iter", type=int, default=4096)
    ap.add_argument("--epochs-per-iter", type=int, default=2)
    ap.add_argument("--minibatch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epsilon", type=float, default=0.2, help="Fraction of non-teacher moves played")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--log-dir", type=str, default="logs")
    ap.add_argument("--out", type=str, default=None, help="Checkpoint path for the best policy")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, log_name="ppo_bc")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    env = make_env(mode=args.mode, num_players=args.num_players)
    env.reset()
    obs, _, _ = env.get_obs()
    in_channels = obs.shape[-1]

    if args.resume:
        policy, meta, _ = load_checkpoint(
            args.resume,
            lambda m: PPOPolicy(in_channels=int(m.get("in_channels", in_channels))),
            device=device,
        )
        logger.info("Resumed from %s (%s)", args.resume, meta)
    else:
        policy = PPOPolicy(in_channels=in_channels).to(device)

    teacher = get_bot_by_name(args.teacher)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(args.log_dir, "checkpoints", f"ppo_bc_{run_id}.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Reuse the PPO loop only for its evaluation harness.
    evaluator = PPOTrainingLoop(
        env=env,
        policy=policy,
        metrics=PPOMetricsLogger(logger),
        device=device,
        opponents=[bot_opponent(get_bot_by_name(n)) for n in BENCHMARK_LEAGUE],
    )

    meta = {
        "algorithm": "ppo",
        "in_channels": in_channels,
        "num_actions": 4,
        "mode": args.mode,
        "num_players": args.num_players,
        "run_id": run_id,
        "pretrain": {"kind": "behaviour_cloning", "teacher": args.teacher, "epsilon": args.epsilon},
    }
    best_points = float("-inf")

    for it in range(1, args.iterations + 1):
        obs_arr, actions, masks, returns = collect(
            env, teacher, policy, device, args.steps_per_iter, args.epsilon, args.gamma
        )
        acts_t = torch.as_tensor(actions, device=device)
        masks_t = torch.as_tensor(masks, device=device)
        rets_t = torch.as_tensor(returns, device=device)

        n = len(actions)
        policy.train()
        tot_loss = tot_acc = tot_v = 0.0
        updates = 0
        for _ in range(args.epochs_per_iter):
            perm = np.random.permutation(n)
            for start in range(0, n, args.minibatch_size):
                idx = perm[start : start + args.minibatch_size]
                if len(idx) < 2:
                    continue
                features = policy._features(obs_arr[idx])
                logits = policy.actor(features)
                logits = logits.masked_fill(~masks_t[idx].bool(), -1e9)
                values = policy.critic(features).squeeze(-1)

                ce = nn.functional.cross_entropy(logits, acts_t[idx])
                v_loss = nn.functional.mse_loss(values, rets_t[idx])
                loss = ce + args.value_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

                tot_loss += float(ce.item())
                tot_v += float(v_loss.item())
                tot_acc += float((logits.argmax(dim=-1) == acts_t[idx]).float().mean().item())
                updates += 1

        logger.info(
            "BC iter %s/%s | samples=%s | ce=%.4f | value=%.4f | teacher_match=%.1f%%",
            it, args.iterations, n, tot_loss / max(updates, 1), tot_v / max(updates, 1),
            100.0 * tot_acc / max(updates, 1),
        )

        if args.eval_every > 0 and it % args.eval_every == 0:
            stats = evaluator.evaluate_policy(args.eval_episodes)
            logger.info(
                "BC eval iter %s | win_rate=%.1f%% | avg_points=%.3f | avg_steps=%.1f | avg_len=%.1f",
                it, stats["win_rate"] * 100.0, stats["avg_points"], stats["avg_steps"], stats["avg_len"],
            )
            if stats["avg_points"] > best_points:
                best_points = stats["avg_points"]
                save_checkpoint(out_path, policy, {**meta, "benchmark": stats, "iteration": it})
                logger.info("Saved best BC policy -> %s (avg_points=%.3f)", out_path, best_points)

    if best_points == float("-inf"):
        save_checkpoint(out_path, policy, meta)
        logger.info("Saved BC policy -> %s", out_path)


if __name__ == "__main__":
    main()
