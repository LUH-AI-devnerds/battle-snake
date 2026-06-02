import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.training.checkpoint import default_checkpoint_dir, save_checkpoint
from battlesnake_ai.training.logger import setup_logger
from battlesnake_ai.training.ppo_loop import PPOTrainingLoop, PPOMetricsLogger
from battlesnake_ai.viz.board_gui import BoardGUI


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Battlesnake PPO with hisss")
    parser.add_argument("--mode", type=str, default="duel", choices=["duel", "standard", "restricted_duel", "restricted_standard"])
    parser.add_argument("--num-players", type=int, default=None, metavar="N")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, log_name="ppo_train")

    env = make_env(mode=args.mode, num_players=args.num_players)
    try:
        logger.info("hisss reward_cfg: %s", asdict(env.cfg.reward_cfg))
    except Exception:
        pass
    env.reset()
    obs, _, _ = env.get_obs()
    in_channels = obs.shape[-1]

    policy = PPOPolicy(in_channels=in_channels)
    metrics = PPOMetricsLogger(logger)
    gui = BoardGUI(title=f"Battlesnake PPO — {args.mode}") if args.gui else None

    loop = PPOTrainingLoop(
        env=env,
        policy=policy,
        metrics=metrics,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        gui=gui,
        gui_every=args.gui_every,
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
            "algorithm": "ppo",
            "in_channels": in_channels,
            "num_actions": 4,
            "mode": args.mode,
            "num_players": args.num_players,
            "hyperparams": {
                "gamma": args.gamma,
                "rollout_steps": args.rollout_steps,
                "clip_eps": args.clip_eps,
            },
        }

    def save_policy(tag: str) -> None:
        path = os.path.join(ckpt_dir, f"ppo_{run_id}_{tag}.pt")
        save_checkpoint(path, policy, build_meta())
        logger.info("Saved checkpoint %s", path)

    def on_episode_end(ep: int) -> None:
        if args.checkpoint_every > 0 and ep % args.checkpoint_every == 0:
            save_policy(f"ep{ep}")

    try:
        loop.train(num_episodes=args.episodes, on_episode_end=on_episode_end)
        save_policy("final")
        save_checkpoint(os.path.join(ckpt_dir, "ppo_latest.pt"), policy, build_meta())
    finally:
        if gui is not None:
            gui.close()


if __name__ == "__main__":
    main()
