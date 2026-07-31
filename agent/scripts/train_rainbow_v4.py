"""Train Battlesnake Rainbow DQN v4 — improved training loop.

Key improvements over v2:
  - Soft target updates (Polyak τ=0.005) instead of hard copy every N steps
  - Cosine-annealing LR (6e-5 → 1e-5) with warm restarts every 5000 episodes
  - Larger replay buffer (500k), bigger batch (512)
  - Simplified reward shaping: living bonus + kill bonus only
  - Fresh optimizer on resume (no stale Adam momentum)
  - Fixed 60% self-play fraction (no curriculum ramp)
  - Dual evaluation: pool win rate AND random win rate
  - train_every=4 to reduce over-fitting to stale replay data
  - n_step=3 (less noisy than 5 in 4-player games)
"""

import argparse
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_sched

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.models.rainbow_dqn import RainbowDQN
from battlesnake_ai.training.checkpoint import default_checkpoint_dir, save_checkpoint
from battlesnake_ai.training.dqn_logging import DQNMetricsLogger
from battlesnake_ai.training.logger import get_tensorboard_writer, setup_logger
from battlesnake_ai.training.prioritized_replay import PrioritizedReplayBuffer
from battlesnake_ai.training.rainbow_loop import RainbowTrainingLoop
from battlesnake_ai.training.opponent_pool import OpponentPool


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Battlesnake Rainbow DQN v4")
    parser.add_argument(
        "--mode",
        type=str,
        default="restricted_standard",
        choices=["duel", "standard", "restricted_duel", "restricted_standard"],
    )
    parser.add_argument("--num-players", type=int, default=None, metavar="N")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--replay-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--lr-min", type=float, default=1e-5, help="Min LR for cosine annealing")
    parser.add_argument("--lr-restart-every", type=int, default=5000, help="Cosine restart period (episodes)")
    parser.add_argument("--train-after", type=int, default=2000)
    parser.add_argument("--train-every", type=int, default=4, help="Run an optimizer step every N env steps")
    parser.add_argument("--soft-target-tau", type=float, default=0.005,
                        help="Polyak averaging coefficient for soft target updates (0 = hard copy)")
    parser.add_argument("--target-update-every", type=int, default=1000,
                        help="Hard target update interval (only used if soft-target-tau=0)")
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--epsilon-start", type=float, default=0.3,
                        help="Start epsilon (lower since resuming from trained model)")
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    parser.add_argument("--n-step", type=int, default=3, help="n-step return horizon")
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-1.0, help="C51 support minimum")
    parser.add_argument("--v-max", type=float, default=1.0, help="C51 support maximum")
    parser.add_argument("--feature-dim", type=int, default=128, help="CNN backbone feature dimension")
    parser.add_argument("--noisy", action="store_true", default=True, help="Use NoisyNet exploration")
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--beta-start", type=float, default=0.4)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--beta-anneal-steps", type=int, default=200_000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-every", type=int, default=1)
    parser.add_argument(
        "--eval-every", type=int, default=50,
        help="Evaluate policy every N episodes",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--self-eval-every", type=int, default=0)
    parser.add_argument("--self-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--self-play", action="store_true", default=True,
        help="Enable self-play with opponent pool",
    )
    parser.add_argument(
        "--self-play-fraction", type=float, default=0.6,
        help="Fixed fraction of episodes that use self-play (no curriculum ramp)",
    )
    parser.add_argument(
        "--pool-snapshot-every", type=int, default=200,
        help="Add model snapshot to opponent pool every N episodes",
    )
    parser.add_argument(
        "--tournament-every", type=int, default=500,
        help="Run pool tournament every N episodes to prune weak NN entries",
    )
    parser.add_argument(
        "--quality-gate-games", type=int, default=20,
        help="Number of games for snapshot quality gate",
    )
    parser.add_argument(
        "--quality-gate-min-wr", type=float, default=0.20,
        help="Minimum win rate vs pool for snapshot admission",
    )
    parser.add_argument(
        "--survival-shaping", action="store_true", default=True,
        help="Enable reward shaping",
    )
    parser.add_argument(
        "--shaping-mode", type=str, default="simplified",
        choices=["simplified", "aggressive", "defensive"],
        help="Reward shaping mode: simplified (kill+live only) or aggressive/defensive (full)",
    )
    parser.add_argument(
        "--survival-strategy", type=str, default="aggressive",
        choices=["aggressive", "defensive"],
    )
    parser.add_argument("--living-bonus", type=float, default=0.005)
    parser.add_argument("--kill-bonus", type=float, default=0.3)
    parser.add_argument("--length-penalty", type=float, default=0.01)
    parser.add_argument("--proximity-penalty", type=float, default=0.005)
    parser.add_argument("--log-updates-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    parser.add_argument("--reset-optimizer", action="store_true", default=True,
                        help="Reset optimizer on resume (don't load stale Adam momentum)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, log_name="rainbow_train_v4")
    tb_writer = get_tensorboard_writer(log_dir=os.path.join(args.log_dir, "tensorboard"))

    env = make_env(mode=args.mode, num_players=args.num_players)
    try:
        logger.info("hisss reward_cfg: %s", asdict(env.cfg.reward_cfg))
    except Exception:
        pass
    env.reset()
    obs, _, _ = env.get_obs()
    in_channels = obs.shape[-1]

    policy = RainbowDQN(
        in_channels=in_channels,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        feature_dim=args.feature_dim,
        noisy=args.noisy,
    )
    target = RainbowDQN(
        in_channels=in_channels,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        feature_dim=args.feature_dim,
        noisy=args.noisy,
    )

    resume_payload: Optional[dict] = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        policy.load_state_dict(state)
        resume_payload = ckpt
        logger.info("Resumed model weights from checkpoint %s", args.resume)
        if args.reset_optimizer:
            logger.info("Optimizer state will NOT be loaded (--reset-optimizer is set)")

    replay = PrioritizedReplayBuffer(capacity=args.replay_size, alpha=args.per_alpha)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    opponent_pool = (
        OpponentPool(
            max_size=20,
            device=device,
            in_channels=in_channels,
            include_baseline_algos=True,
        )
        if args.self_play
        else None
    )
    if opponent_pool is not None:
        # Pre-load historical best NN checkpoints into pool if available
        historical_ckpts = [
            ("best_checkpoint/rainbow_v2_best.pt", "RainbowDQN_v2_best"),
            ("best_checkpoint/rainbow_20260715_214528_best.pt", "RainbowDQN_v1_best"),
            ("best_checkpoint/rainbow_20260704_125842_ep1600.pt", "RainbowDQN_v0_ep1600"),
        ]
        for ckpt_path, label in historical_ckpts:
            if os.path.exists(ckpt_path):
                opponent_pool.add_checkpoint(ckpt_path, label=label, permanent=True)

        logger.info(
            "Opponent pool initialized with NN models: %s",
            opponent_pool.summary().replace("\n", " | "),
        )

    metrics = DQNMetricsLogger(logger=logger, log_dir=args.log_dir, tensorboard_writer=tb_writer)
    gui = None  # No GUI for headless training

    loop = RainbowTrainingLoop(
        env=env,
        policy_net=policy,
        target_net=target,
        replay=replay,
        metrics=metrics,
        gamma=args.gamma,
        n_step=args.n_step,
        lr=args.lr,
        batch_size=args.batch_size,
        train_after=args.train_after,
        train_every=args.train_every,
        target_update_every=args.target_update_every,
        soft_target_tau=args.soft_target_tau,
        max_grad_norm=args.max_grad_norm,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_anneal_steps=args.beta_anneal_steps,
        gui=gui,
        gui_every=args.gui_every,
        console_log_every=args.log_updates_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        self_eval_every=args.self_eval_every,
        self_eval_episodes=args.self_eval_episodes,
        survival_shaping=args.survival_shaping,
        shaping_mode=args.shaping_mode,
        living_bonus=args.living_bonus,
        kill_bonus=args.kill_bonus,
        length_penalty=args.length_penalty,
        proximity_penalty=args.proximity_penalty,
        survival_strategy=args.survival_strategy,
        opponent_pool=opponent_pool,
        self_play_fraction=args.self_play_fraction,
    )

    # Set up cosine annealing LR scheduler with warm restarts
    # T_0 is in optimizer steps, not episodes. Estimate ~150 env steps/episode,
    # with train_every=4 that's ~37 optimizer steps/episode.
    steps_per_episode_est = 37  # conservative estimate
    t0_steps = args.lr_restart_every * steps_per_episode_est
    scheduler = lr_sched.CosineAnnealingWarmRestarts(
        loop.optimizer,
        T_0=max(t0_steps, 1000),
        T_mult=1,
        eta_min=args.lr_min,
    )
    loop.lr_scheduler = scheduler
    logger.info(
        "LR scheduler: CosineAnnealingWarmRestarts T_0=%d steps (~%d episodes), eta_min=%s",
        t0_steps, args.lr_restart_every, args.lr_min,
    )

    if resume_payload is not None:
        load_opt = not args.reset_optimizer
        loop.load_training_state(resume_payload, load_optimizer=load_opt)
        logger.info(
            "Restored training state: total_env_steps=%s optim_steps=%s best_win_rate=%.4f best_episode=%s",
            loop.total_env_steps,
            loop.optim_steps,
            loop.best_win_rate,
            loop.best_episode,
        )

    ckpt_dir = (
        os.path.abspath(args.checkpoint_dir)
        if args.checkpoint_dir
        else str(default_checkpoint_dir(args.log_dir))
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def build_meta() -> dict:
        return {
            "algorithm": "rainbow",
            "in_channels": in_channels,
            "num_actions": 4,
            "mode": args.mode,
            "num_players": args.num_players,
            "num_atoms": args.num_atoms,
            "v_min": args.v_min,
            "v_max": args.v_max,
            "feature_dim": args.feature_dim,
            "noisy": args.noisy,
            "run_id": run_id,
            "version": "v4",
            "total_episodes": args.episodes,
            "seed": args.seed,
            "hyperparams": {
                "gamma": args.gamma,
                "n_step": args.n_step,
                "lr": args.lr,
                "lr_min": args.lr_min,
                "lr_restart_every": args.lr_restart_every,
                "per_alpha": args.per_alpha,
                "batch_size": args.batch_size,
                "replay_size": args.replay_size,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
                "epsilon_decay_steps": args.epsilon_decay_steps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
                "beta_anneal_steps": args.beta_anneal_steps,
                "train_after": args.train_after,
                "train_every": args.train_every,
                "soft_target_tau": args.soft_target_tau,
                "target_update_every": args.target_update_every,
                "max_grad_norm": args.max_grad_norm,
                "eval_episodes": args.eval_episodes,
                "eval_seed": args.eval_seed,
                "survival_shaping": args.survival_shaping,
                "shaping_mode": args.shaping_mode,
                "survival_strategy": args.survival_strategy,
                "living_bonus": args.living_bonus,
                "kill_bonus": args.kill_bonus,
                "length_penalty": args.length_penalty,
                "proximity_penalty": args.proximity_penalty,
                "self_play": args.self_play,
                "self_play_fraction": args.self_play_fraction,
                "pool_snapshot_every": args.pool_snapshot_every,
                "quality_gate_min_wr": args.quality_gate_min_wr,
                "reset_optimizer": args.reset_optimizer,
            },
        }

    def save_policy(tag: str, *, include_training_state: bool = True) -> None:
        path = os.path.join(ckpt_dir, f"rainbow_v4_{run_id}_{tag}.pt")
        save_checkpoint(
            path,
            policy,
            build_meta(),
            optimizer=loop.optimizer,
            training_state=loop.get_training_state() if include_training_state else None,
        )
        logger.info("Saved checkpoint %s", path)

    def save_best(tag_prefix: str = "best") -> None:
        tagged = os.path.join(ckpt_dir, f"rainbow_v4_{run_id}_{tag_prefix}.pt")
        save_checkpoint(
            tagged,
            policy,
            build_meta(),
            optimizer=loop.optimizer,
            training_state=loop.get_training_state(),
        )
        alias = os.path.join(ckpt_dir, f"rainbow_v4_{tag_prefix}.pt")
        save_checkpoint(
            alias,
            policy,
            build_meta(),
            optimizer=loop.optimizer,
            training_state=loop.get_training_state(),
        )
        logger.info(
            "Saved %s checkpoint (win_rate=%.4f at ep %s) -> %s, %s",
            tag_prefix,
            loop.best_win_rate,
            loop.best_episode,
            tagged,
            alias,
        )

    # Compute episode offset for resume
    _resumed_episodes = 0
    if args.resume and resume_payload is not None:
        ts = resume_payload.get("training_state", resume_payload)
        _resumed_episodes = int(ts.get("best_episode", 0))
        import re
        m = re.search(r"ep(\d+)", args.resume)
        if m:
            _resumed_episodes = int(m.group(1))
        logger.info("Episode offset from resume: %d", _resumed_episodes)

    def on_episode_end(ep: int) -> None:
        global_ep = ep + _resumed_episodes

        if args.checkpoint_every > 0 and ep % args.checkpoint_every == 0:
            save_policy(f"ep{global_ep}")

        if args.self_play and opponent_pool is not None:
            # Fixed self-play fraction (no curriculum ramp)
            loop.self_play_fraction = args.self_play_fraction

            if ep % args.pool_snapshot_every == 0 and ep > 0:
                # Quality gate: only add snapshot if it passes win rate threshold vs NN pool
                if opponent_pool.quality_gate(
                    policy,
                    env,
                    num_games=args.quality_gate_games,
                    min_win_rate=args.quality_gate_min_wr,
                ):
                    opponent_pool.add_snapshot(policy, label=f"v4_ep{global_ep}")
                    logger.info(
                        "Added quality-gated NN snapshot to pool (total pool size: %d NN models)",
                        opponent_pool.size,
                    )
                else:
                    logger.info("Snapshot at ep%d failed quality gate — not added to pool", global_ep)

            # Tournament pruning
            if args.tournament_every > 0 and ep % args.tournament_every == 0 and ep > 0:
                evicted = opponent_pool.run_tournament(env, policy)
                if evicted:
                    logger.info("NN Tournament evicted %d entries: %s", len(evicted), evicted)
                logger.info("NN Pool status: %s", opponent_pool.summary().replace('\n', ' | '))

    def on_eval(ep: int, eval_stats: Dict[str, float], is_best: bool) -> None:
        if is_best:
            save_best("best_pool")

        # Also evaluate against random opponents for an absolute-strength metric
        random_stats = loop.evaluate_vs_random(num_episodes=args.eval_episodes)
        logger.info(
            "Evaluation_vs_random at Episode %s | win_rate=%.1f%% | avg_steps=%.1f | avg_return=%.4f",
            ep,
            random_stats["win_rate"] * 100.0,
            random_stats["avg_steps"],
            random_stats["avg_return"],
        )
        if random_stats["win_rate"] > loop.best_random_win_rate:
            loop.best_random_win_rate = random_stats["win_rate"]
            save_best("best_random")
            logger.info(
                "New best random win rate: %.1f%% at episode %s",
                random_stats["win_rate"] * 100.0,
                ep,
            )

        # Log current LR
        current_lr = loop.optimizer.param_groups[0]["lr"]
        logger.info("Current LR: %.2e | optim_steps: %d", current_lr, loop.optim_steps)

    try:
        loop.train(
            num_episodes=args.episodes,
            on_episode_end=on_episode_end,
            on_eval=on_eval,
        )
        save_policy("final")
        save_checkpoint(
            os.path.join(ckpt_dir, "rainbow_v4_latest.pt"),
            policy,
            build_meta(),
            optimizer=loop.optimizer,
            training_state=loop.get_training_state(),
        )
        logger.info(
            "Run complete | best_pool_win_rate=%.4f | best_random_win_rate=%.4f at episode %s",
            loop.best_win_rate,
            loop.best_random_win_rate,
            loop.best_episode,
        )
    finally:
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
