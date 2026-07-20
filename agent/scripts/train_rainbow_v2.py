import argparse
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.models.rainbow_dqn import RainbowDQN
from battlesnake_ai.training.checkpoint import default_checkpoint_dir, save_checkpoint
from battlesnake_ai.training.dqn_logging import DQNMetricsLogger
from battlesnake_ai.training.logger import get_tensorboard_writer, setup_logger
from battlesnake_ai.training.prioritized_replay import PrioritizedReplayBuffer
from battlesnake_ai.training.rainbow_loop import RainbowTrainingLoop
from battlesnake_ai.training.opponent_pool import OpponentPool
from battlesnake_ai.viz.board_gui import BoardGUI


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Battlesnake Rainbow DQN v2 with Self-Play")
    parser.add_argument(
        "--mode",
        type=str,
        default="duel",
        choices=["duel", "standard", "restricted_duel", "restricted_standard"],
    )
    parser.add_argument("--num-players", type=int, default=None, metavar="N")
    parser.add_argument("--episodes", type=int, default=15000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--replay-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train-after", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=1, help="Run an optimizer step every N env steps")
    parser.add_argument("--target-update-every", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--n-step", type=int, default=5, help="n-step return horizon")
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
        "--eval-every",
        type=int,
        default=25,
        help="Evaluate policy against random every N episodes",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--self-eval-every",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--self-eval-episodes",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        default=True,
        help="Enable self-play curriculum",
    )
    parser.add_argument(
        "--pool-snapshot-every",
        type=int,
        default=50,
        help="Add model snapshot to opponent pool every N episodes",
    )
    parser.add_argument(
        "--survival-shaping",
        action="store_true",
        default=True,
        help="Reward shaping for grow-then-hunt",
    )
    parser.add_argument(
        "--survival-strategy",
        type=str,
        default="aggressive",
        choices=["aggressive", "defensive"],
    )
    parser.add_argument("--living-bonus", type=float, default=0.01)
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.05,
    )
    parser.add_argument("--proximity-penalty", type=float, default=0.02)
    parser.add_argument("--log-updates-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, log_name="rainbow_train_v2")
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
        logger.info("Resumed from checkpoint %s", args.resume)

    replay = PrioritizedReplayBuffer(capacity=args.replay_size, alpha=args.per_alpha)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opponent_pool = OpponentPool(max_size=20, device=device) if args.self_play else None

    metrics = DQNMetricsLogger(logger=logger, log_dir=args.log_dir, tensorboard_writer=tb_writer)
    gui = BoardGUI(title=f"Battlesnake Rainbow v2 — {args.mode}") if args.gui else None

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
        living_bonus=args.living_bonus,
        length_penalty=args.length_penalty,
        proximity_penalty=args.proximity_penalty,
        survival_strategy=args.survival_strategy,
        opponent_pool=opponent_pool,
        self_play_fraction=0.0,  # Updated dynamically
    )

    if resume_payload is not None:
        loop.load_training_state(resume_payload, load_optimizer=True)
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
            "total_episodes": args.episodes,
            "seed": args.seed,
            "hyperparams": {
                "gamma": args.gamma,
                "n_step": args.n_step,
                "lr": args.lr,
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
                "target_update_every": args.target_update_every,
                "max_grad_norm": args.max_grad_norm,
                "eval_episodes": args.eval_episodes,
                "eval_seed": args.eval_seed,
                "self_eval_every": args.self_eval_every,
                "self_eval_episodes": args.self_eval_episodes,
                "survival_shaping": args.survival_shaping,
                "survival_strategy": args.survival_strategy,
                "living_bonus": args.living_bonus,
                "length_penalty": args.length_penalty,
                "proximity_penalty": args.proximity_penalty,
                "self_play": args.self_play,
            },
        }

    def save_policy(tag: str, *, include_training_state: bool = True) -> None:
        path = os.path.join(ckpt_dir, f"rainbow_v2_{run_id}_{tag}.pt")
        save_checkpoint(
            path,
            policy,
            build_meta(),
            optimizer=loop.optimizer,
            training_state=loop.get_training_state() if include_training_state else None,
        )
        logger.info("Saved checkpoint %s", path)

    def save_best() -> None:
        tagged = os.path.join(ckpt_dir, f"rainbow_v2_{run_id}_best.pt")
        save_checkpoint(
            tagged,
            policy,
            build_meta(),
            optimizer=loop.optimizer,
            training_state=loop.get_training_state(),
        )
        alias = os.path.join(ckpt_dir, "rainbow_v2_best.pt")
        save_checkpoint(
            alias,
            policy,
            build_meta(),
            optimizer=loop.optimizer,
            training_state=loop.get_training_state(),
        )
        logger.info(
            "Saved best checkpoint (win_rate=%.4f at ep %s) -> %s, %s",
            loop.best_win_rate,
            loop.best_episode,
            tagged,
            alias,
        )

    # Compute episode offset so self-play curriculum continues correctly on resume.
    # total_env_steps was restored from the checkpoint; approximate the episode
    # number from it (episodes ≈ steps / avg_steps_per_ep, but we can use the
    # checkpoint's episode count directly from training_state if available).
    _resumed_episodes = 0
    if args.resume and resume_payload is not None:
        ts = resume_payload.get("training_state", resume_payload)
        _resumed_episodes = int(ts.get("best_episode", 0))
        # Use the checkpoint tag (ep8500) as a better estimate if available
        import re
        m = re.search(r"ep(\d+)", args.resume)
        if m:
            _resumed_episodes = int(m.group(1))
        logger.info("Self-play curriculum offset: %d episodes from resume", _resumed_episodes)

    def on_episode_end(ep: int) -> None:
        global_ep = ep + _resumed_episodes  # Account for episodes before resume

        if args.checkpoint_every > 0 and ep % args.checkpoint_every == 0:
            save_policy(f"ep{global_ep}")
            
        if args.self_play:
            # Curriculum: peak at 80% self-play at half total episodes
            total_target = args.episodes + _resumed_episodes
            frac = min(0.8, (global_ep / (total_target / 2)) * 0.8)
            loop.self_play_fraction = frac
            
            if ep % args.pool_snapshot_every == 0:
                opponent_pool.add_snapshot(policy)
                logger.info(f"Added snapshot to opponent pool (size: {len(opponent_pool.snapshots)})")

    def on_eval(ep: int, eval_stats: Dict[str, float], is_best: bool) -> None:
        if is_best:
            save_best()

    try:
        loop.train(
            num_episodes=args.episodes,
            on_episode_end=on_episode_end,
            on_eval=on_eval,
        )
        save_policy("final")
        save_checkpoint(
            os.path.join(ckpt_dir, "rainbow_latest.pt"),
            policy,
            build_meta(),
            optimizer=loop.optimizer,
            training_state=loop.get_training_state(),
        )
        logger.info(
            "Run complete | best_win_rate=%.4f at episode %s",
            loop.best_win_rate,
            loop.best_episode,
        )
    finally:
        if gui is not None:
            gui.close()
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
