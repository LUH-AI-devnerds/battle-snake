import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.models.dqn import DQN
from battlesnake_ai.training.checkpoint import default_checkpoint_dir, save_checkpoint
from battlesnake_ai.training.dqn_logging import DQNMetricsLogger
from battlesnake_ai.training.dqn_loop import DQNTrainingLoop
from battlesnake_ai.training.logger import get_tensorboard_writer, setup_logger
from battlesnake_ai.training.replay_buffer import ReplayBuffer
from battlesnake_ai.blackout.client import sample_leaderboard_game_ids
from battlesnake_ai.viz.blackout_replay import BlackoutReplayViewer
from battlesnake_ai.viz.board_gui import BoardGUI


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Battlesnake DQN with hisss")
    parser.add_argument(
        "--mode",
        type=str,
        default="restricted_standard",
        choices=["duel", "standard", "restricted_duel", "restricted_standard"],
        help="Game mode",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=None,
        metavar="N",
        help="Number of snakes (default: 2 for duel modes, 4 for standard modes)",
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
        "--eval-every",
        type=int,
        default=10,
        help="Evaluate policy against random every N episodes (0 to disable)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--log-updates-every",
        type=int,
        default=10,
        help="Print DQN update lines to the console every N gradient steps (JSONL logs all steps)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save policy checkpoint every N episodes (0 = only at end)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (default: <log-dir>/checkpoints)",
    )
    parser.add_argument(
        "--blackout-replay",
        type=int,
        default=None,
        metavar="GAME_ID",
        help="Watch a Blackout leaderboard replay (game id) before training",
    )
    parser.add_argument(
        "--blackout-replay-every",
        type=int,
        default=0,
        metavar="N",
        help="During training with --gui, replay a random leaderboard game every N episodes (0=off)",
    )
    parser.add_argument(
        "--blackout-replay-fps",
        type=float,
        default=6.0,
        help="Playback FPS for Blackout replays during training",
    )

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, log_name="dqn_train")
    tb_writer = get_tensorboard_writer(log_dir=os.path.join(args.log_dir, "tensorboard"))

    logger.info(
        "Building environment (%s%s)...",
        args.mode,
        f", num_players={args.num_players}" if args.num_players is not None else "",
    )
    env = make_env(mode=args.mode, num_players=args.num_players)
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
    blackout_viewer = BlackoutReplayViewer(gui=gui) if args.gui else None
    blackout_pool = None
    if args.blackout_replay_every > 0 and args.gui:
        logger.info("Fetching recent Blackout leaderboard games for replay...")
        blackout_pool = sample_leaderboard_game_ids()
        logger.info("Loaded %s replay candidates from bs-blackout-2026", len(blackout_pool))

    if args.blackout_replay is not None:
        logger.info("Watching Blackout replay game %s...", args.blackout_replay)
        viewer = blackout_viewer or BlackoutReplayViewer()
        try:
            viewer.play_game_id(args.blackout_replay, fps=args.blackout_replay_fps)
        finally:
            if blackout_viewer is None:
                viewer.close()

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
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
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

    ckpt_dir = (
        os.path.abspath(args.checkpoint_dir)
        if args.checkpoint_dir
        else str(default_checkpoint_dir(args.log_dir))
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def build_meta() -> dict:
        return {
            "algorithm": "dqn",
            "in_channels": in_channels,
            "num_actions": 4,
            "mode": args.mode,
            "num_players": args.num_players,
            "hyperparams": {
                "gamma": args.gamma,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epsilon_decay_steps": args.epsilon_decay_steps,
            },
        }

    def save_policy(tag: str) -> str:
        path = os.path.join(ckpt_dir, f"dqn_{run_id}_{tag}.pt")
        save_checkpoint(path, policy, build_meta())
        logger.info("Saved checkpoint %s", path)
        return path

    def on_episode_end(ep: int) -> None:
        if args.checkpoint_every > 0 and ep % args.checkpoint_every == 0:
            save_policy(f"ep{ep}")
        if (
            args.blackout_replay_every > 0
            and blackout_viewer is not None
            and blackout_pool
            and ep % args.blackout_replay_every == 0
        ):
            gid = blackout_viewer.play_random_leaderboard(
                fps=args.blackout_replay_fps,
                pool=blackout_pool,
            )
            if gid is not None:
                logger.info("Replayed Blackout game %s", gid)

    try:
        loop.train(num_episodes=args.episodes, on_episode_end=on_episode_end)
        save_policy("final")
        latest = os.path.join(ckpt_dir, "dqn_latest.pt")
        save_checkpoint(latest, policy, build_meta())
        logger.info("Saved latest checkpoint %s", latest)
    finally:
        if gui is not None:
            gui.close()
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
