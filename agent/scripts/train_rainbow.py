import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.models.rainbow_dqn import RainbowDQN
from battlesnake_ai.training.checkpoint import default_checkpoint_dir, save_checkpoint
from battlesnake_ai.training.dqn_logging import DQNMetricsLogger
from battlesnake_ai.training.logger import get_tensorboard_writer, setup_logger
from battlesnake_ai.training.prioritized_replay import PrioritizedReplayBuffer
from battlesnake_ai.training.rainbow_loop import RainbowTrainingLoop
from battlesnake_ai.viz.board_gui import BoardGUI


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Battlesnake Rainbow DQN with hisss")
    parser.add_argument(
        "--mode",
        type=str,
        default="duel",
        choices=["duel", "standard", "restricted_duel", "restricted_standard"],
    )
    parser.add_argument("--num-players", type=int, default=None, metavar="N")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-after", type=int, default=500)
    parser.add_argument("--target-update-every", type=int, default=500)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--n-step", type=int, default=3, help="n-step discount exponent on bootstrap")
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--beta-start", type=float, default=0.4)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--beta-anneal-steps", type=int, default=50_000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-every", type=int, default=1)
    parser.add_argument("--log-updates-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, log_name="rainbow_train")
    tb_writer = get_tensorboard_writer(log_dir=os.path.join(args.log_dir, "tensorboard"))

    env = make_env(mode=args.mode, num_players=args.num_players)
    try:
        logger.info("hisss reward_cfg: %s", asdict(env.cfg.reward_cfg))
    except Exception:
        pass
    env.reset()
    obs, _, _ = env.get_obs()
    in_channels = obs.shape[-1]

    policy = RainbowDQN(in_channels=in_channels, num_atoms=args.num_atoms)
    target = RainbowDQN(in_channels=in_channels, num_atoms=args.num_atoms)
    replay = PrioritizedReplayBuffer(capacity=args.replay_size, alpha=args.per_alpha)

    metrics = DQNMetricsLogger(logger=logger, log_dir=args.log_dir, tensorboard_writer=tb_writer)
    gui = BoardGUI(title=f"Battlesnake Rainbow — {args.mode}") if args.gui else None

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
        target_update_every=args.target_update_every,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_anneal_steps=args.beta_anneal_steps,
        gui=gui,
        gui_every=args.gui_every,
        console_log_every=args.log_updates_every,
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
            "hyperparams": {
                "gamma": args.gamma,
                "n_step": args.n_step,
                "lr": args.lr,
                "per_alpha": args.per_alpha,
            },
        }

    def save_policy(tag: str) -> None:
        path = os.path.join(ckpt_dir, f"rainbow_{run_id}_{tag}.pt")
        save_checkpoint(path, policy, build_meta())
        logger.info("Saved checkpoint %s", path)

    def on_episode_end(ep: int) -> None:
        if args.checkpoint_every > 0 and ep % args.checkpoint_every == 0:
            save_policy(f"ep{ep}")

    try:
        loop.train(num_episodes=args.episodes, on_episode_end=on_episode_end)
        save_policy("final")
        save_checkpoint(os.path.join(ckpt_dir, "rainbow_latest.pt"), policy, build_meta())
    finally:
        if gui is not None:
            gui.close()
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
