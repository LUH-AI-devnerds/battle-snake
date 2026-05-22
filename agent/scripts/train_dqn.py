import argparse
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.models.dqn import DQN
from battlesnake_ai.training.dqn_logging import DQNMetricsLogger
from battlesnake_ai.training.dqn_loop import DQNTrainingLoop
from battlesnake_ai.training.logger import get_tensorboard_writer, setup_logger
from battlesnake_ai.training.replay_buffer import ReplayBuffer
from battlesnake_ai.viz.board_gui import BoardGUI


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Battlesnake DQN with hisss")
    parser.add_argument(
        "--mode",
        type=str,
        default="restricted_standard",
        choices=["duel", "standard", "restricted_duel", "restricted_standard"],
        help="Game mode (DQN loop is wired for duel / two snakes)",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs and metrics")
    parser.add_argument("--replay-size", type=int, default=50_000, help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=64, help="DQN minibatch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor γ")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--train-after", type=int, default=500, help="Minimum transitions before updates")
    parser.add_argument("--target-update-every", type=int, default=500, help="Optimizer steps between target sync")
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000, help="Steps to decay ε from start to end")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--gui", action="store_true", help="Open matplotlib board window during training")
    parser.add_argument("--gui-every", type=int, default=1, help="Refresh GUI every N env steps")
    parser.add_argument(
        "--log-updates-every",
        type=int,
        default=10,
        help="Print DQN update lines to the console every N gradient steps (JSONL logs all steps)",
    )

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, log_name="dqn_train")
    tb_writer = get_tensorboard_writer(log_dir=os.path.join(args.log_dir, "tensorboard"))

    logger.info("Building environment (%s)...", args.mode)
    env = make_env(mode=args.mode)
    try:
        logger.info("hisss reward_cfg (how match rewards are shaped): %s", asdict(env.cfg.reward_cfg))
    except Exception:
        logger.info("hisss reward_cfg: %s", env.cfg.reward_cfg)
    env.reset()
    obs, _, _ = env.get_obs()
    in_channels = obs.shape[-1]

    logger.info("Observation shape: %s | channels=%s", obs.shape, in_channels)

    policy = DQN(in_channels=in_channels, num_actions=4)
    target = DQN(in_channels=in_channels, num_actions=4)
    replay = ReplayBuffer(capacity=args.replay_size)

    metrics = DQNMetricsLogger(logger=logger, log_dir=args.log_dir, tensorboard_writer=tb_writer)

    gui = BoardGUI(title=f"Battlesnake DQN — {args.mode}") if args.gui else None

    loop = DQNTrainingLoop(
        env=env,
        policy_net=policy,
        target_net=target,
        replay=replay,
        metrics=metrics,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        train_after=args.train_after,
        target_update_every=args.target_update_every,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        gui=gui,
        gui_every=args.gui_every,
        console_log_every=args.log_updates_every,
    )

    if gui is not None:
        gui.set_run_metadata(
            mode=args.mode,
            obs_shape=tuple(obs.shape),
            in_channels=in_channels,
            log_dir=os.path.abspath(args.log_dir),
            gamma=args.gamma,
            lr=args.lr,
            batch_size=args.batch_size,
            train_after=args.train_after,
            target_update_every=args.target_update_every,
            epsilon_decay_steps=args.epsilon_decay_steps,
            replay_capacity=args.replay_size,
            device=str(loop.device),
        )
        try:
            gui.set_run_metadata(reward_cfg=asdict(env.cfg.reward_cfg))
        except Exception:
            pass

    try:
        loop.train(num_episodes=args.episodes)
    finally:
        if gui is not None:
            gui.close()
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
