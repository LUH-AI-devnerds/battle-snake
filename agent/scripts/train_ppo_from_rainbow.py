import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.inference.agent_loader import copy_rainbow_backbone_to_ppo, load_agent
from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.models.rainbow_dqn import RainbowDQN
from battlesnake_ai.training.checkpoint import default_checkpoint_dir, save_checkpoint
from battlesnake_ai.training.logger import setup_logger
from battlesnake_ai.training.ppo_loop import PPOTrainingLoop, PPOMetricsLogger
from battlesnake_ai.viz.board_gui import BoardGUI


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PPO from a Rainbow DQN checkpoint")
    parser.add_argument("--rainbow-checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="duel", choices=["duel", "standard", "restricted_duel", "restricted_standard"])
    parser.add_argument("--num-players", type=int, default=None, metavar="N")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--freeze-encoder", action="store_true", help="Train only actor/critic heads")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, log_name="ppo_finetune_train")

    rainbow_model, rainbow_meta = load_agent(args.rainbow_checkpoint)
    if not isinstance(rainbow_model, RainbowDQN):
        raise TypeError("Checkpoint must be a Rainbow DQN model")

    env = make_env(mode=args.mode, num_players=args.num_players)
    env.reset()
    obs, _, _ = env.get_obs()
    in_channels = obs.shape[-1]
    if int(rainbow_meta.get("in_channels", in_channels)) != in_channels:
        logger.warning(
            "Checkpoint in_channels=%s differs from env %s",
            rainbow_meta.get("in_channels"),
            in_channels,
        )

    policy = PPOPolicy(in_channels=in_channels)
    copy_rainbow_backbone_to_ppo(rainbow_model, policy)
    logger.info("Initialized PPO backbone from %s", args.rainbow_checkpoint)

    metrics = PPOMetricsLogger(logger)
    gui = BoardGUI(title=f"PPO fine-tune — {args.mode}") if args.gui else None

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
        gui=gui,
        freeze_encoder=args.freeze_encoder,
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
            "algorithm": "ppo_finetune",
            "init_from": "rainbow",
            "rainbow_checkpoint": os.path.abspath(args.rainbow_checkpoint),
            "in_channels": in_channels,
            "num_actions": 4,
            "mode": args.mode,
            "num_players": args.num_players,
            "freeze_encoder": args.freeze_encoder,
        }

    def save_policy(tag: str) -> None:
        path = os.path.join(ckpt_dir, f"ppo_finetune_{run_id}_{tag}.pt")
        save_checkpoint(path, policy, build_meta())
        logger.info("Saved checkpoint %s", path)

    def on_episode_end(ep: int) -> None:
        if args.checkpoint_every > 0 and ep % args.checkpoint_every == 0:
            save_policy(f"ep{ep}")

    try:
        loop.train(num_episodes=args.episodes, on_episode_end=on_episode_end)
        save_policy("final")
        save_checkpoint(os.path.join(ckpt_dir, "ppo_finetune_latest.pt"), policy, build_meta())
    finally:
        if gui is not None:
            gui.close()


if __name__ == "__main__":
    main()
