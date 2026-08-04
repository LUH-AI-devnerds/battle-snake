"""Train a Battlesnake PPO policy against a league of opponents.

The league matters more than the algorithm here: training only against random
or a single frozen net produced policies that lose to the repo's own heuristic
bots.  Default league = heuristic bots + optional frozen checkpoints + self-play.

Checkpoints tagged ``best`` are gated on a fixed benchmark league so a lucky
100-episode eval cannot promote a weak policy.
"""

import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.inference.agent_loader import load_agent
from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.training.checkpoint import default_checkpoint_dir, load_checkpoint, save_checkpoint
from battlesnake_ai.training.heuristic_opponents import get_bot_by_name
from battlesnake_ai.training.logger import setup_logger
from battlesnake_ai.training.ppo_loop import (
    Opponent,
    PPOMetricsLogger,
    PPOTrainingLoop,
    bot_opponent,
    model_opponent,
    random_opponent,
)
from battlesnake_ai.viz.board_gui import BoardGUI

DEFAULT_LEAGUE = ["flood_fill", "food_chaser", "aggressive_hunter", "cautious"]
BENCHMARK_LEAGUE = ["aggressive_hunter", "cautious", "flood_fill"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Battlesnake PPO with hisss")
    parser.add_argument("--mode", type=str, default="restricted_standard",
                        choices=["duel", "standard", "restricted_duel", "restricted_standard"])
    parser.add_argument("--num-players", type=int, default=4, metavar="N")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-end", type=float, default=3e-5)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.02,
                        help="Stop the epoch loop once approx KL exceeds 1.5x this (0 disables)")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-end", type=float, default=0.002)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Reward shaping
    parser.add_argument("--survival-shaping", action="store_true")
    parser.add_argument("--living-bonus", type=float, default=0.005)
    parser.add_argument("--length-penalty", type=float, default=0.01)
    parser.add_argument("--proximity-penalty", type=float, default=0.005)
    parser.add_argument("--survival-strategy", type=str, default="aggressive",
                        choices=["aggressive", "defensive", "survive"])

    # Opponent league
    parser.add_argument("--league", nargs="*", default=DEFAULT_LEAGUE,
                        help="Heuristic bot names for opponent seats (empty = none)")
    parser.add_argument("--league-checkpoints", nargs="*", default=[],
                        help="Frozen checkpoints (rainbow/dqn/ppo) to add to the league")
    parser.add_argument("--include-random-opponent", action="store_true",
                        help="Add a uniform-random opponent to the league")
    parser.add_argument("--self-play-prob", type=float, default=0.25,
                        help="Fraction of episodes where every seat is the learner")

    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--benchmark-every", type=int, default=500,
                        help="Gate the 'best' checkpoint on the benchmark league every N episodes")
    parser.add_argument("--benchmark-episodes", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None, metavar="PATH")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-freeze-batchnorm", action="store_true",
                        help="Let BatchNorm use batch statistics during training (breaks the "
                             "PPO ratio: collection and update then normalise differently)")

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, log_name="ppo_train")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    env = make_env(mode=args.mode, num_players=args.num_players)
    try:
        logger.info("hisss reward_cfg: %s", asdict(env.cfg.reward_cfg))
    except Exception:
        pass
    env.reset()
    obs, _, _ = env.get_obs()
    in_channels = obs.shape[-1]

    policy = PPOPolicy(in_channels=in_channels)
    if args.resume_checkpoint:
        def _ppo_factory(meta: dict) -> PPOPolicy:
            return PPOPolicy(
                in_channels=int(meta.get("in_channels", in_channels)),
                num_actions=int(meta.get("num_actions", 4)),
            )

        policy, resume_meta, _ = load_checkpoint(args.resume_checkpoint, _ppo_factory, device=device)
        logger.info("Resumed PPO policy from %s (meta=%s)", args.resume_checkpoint, resume_meta)

    opponents: list[Opponent] = [bot_opponent(get_bot_by_name(n)) for n in (args.league or [])]
    for path in args.league_checkpoints:
        model, meta = load_agent(path, device=device)
        opponents.append(model_opponent(os.path.basename(path), model, device))
        logger.info("League opponent %s (meta=%s)", path, meta)
    if args.include_random_opponent:
        opponents.append(random_opponent())

    benchmark = [bot_opponent(get_bot_by_name(n)) for n in BENCHMARK_LEAGUE]

    metrics = PPOMetricsLogger(logger)
    gui = BoardGUI(title=f"Battlesnake PPO — {args.mode}") if args.gui else None

    loop = PPOTrainingLoop(
        env=env,
        policy=policy,
        metrics=metrics,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=args.lr,
        lr_end=args.lr_end,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_eps=args.clip_eps,
        target_kl=args.target_kl,
        entropy_coef=args.entropy_coef,
        entropy_coef_end=args.entropy_end,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        survival_shaping=args.survival_shaping,
        living_bonus=args.living_bonus,
        length_penalty=args.length_penalty,
        proximity_penalty=args.proximity_penalty,
        survival_strategy=args.survival_strategy,
        device=device,
        gui=gui,
        gui_every=args.gui_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        opponents=opponents,
        self_play_prob=args.self_play_prob,
        freeze_batchnorm=not args.no_freeze_batchnorm,
    )

    ckpt_dir = (
        os.path.abspath(args.checkpoint_dir)
        if args.checkpoint_dir
        else str(default_checkpoint_dir(args.log_dir))
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Score the starting policy first. Without this the first benchmark always
    # becomes "best", so a run that degrades a good warm start still publishes
    # its own regression.
    best_points = float("-inf")
    if args.benchmark_every > 0:
        initial = loop.evaluate_policy(args.benchmark_episodes, opponents=benchmark)
        metrics.log_evaluation(0, "benchmark/initial", initial)
        best_points = initial["avg_points"]

    def build_meta(extra: dict | None = None) -> dict:
        meta = {
            "algorithm": "ppo",
            "in_channels": in_channels,
            "num_actions": 4,
            "mode": args.mode,
            "num_players": args.num_players,
            "run_id": run_id,
            "league": [o.name for o in opponents],
            "self_play_prob": args.self_play_prob,
            "hyperparams": {
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "lr": args.lr,
                "rollout_steps": args.rollout_steps,
                "ppo_epochs": args.ppo_epochs,
                "minibatch_size": args.minibatch_size,
                "clip_eps": args.clip_eps,
                "target_kl": args.target_kl,
                "entropy_coef": args.entropy_coef,
                "survival_shaping": args.survival_shaping,
            },
        }
        meta.update(extra or {})
        return meta

    def save_policy(tag: str, extra: dict | None = None) -> None:
        path = os.path.join(ckpt_dir, f"ppo_{run_id}_{tag}.pt")
        save_checkpoint(path, policy, build_meta(extra))
        logger.info("Saved checkpoint %s", path)

    def on_episode_end(ep: int) -> None:
        nonlocal best_points
        if args.checkpoint_every > 0 and ep % args.checkpoint_every == 0:
            save_policy(f"ep{ep}")
        if args.benchmark_every > 0 and ep % args.benchmark_every == 0:
            stats = loop.evaluate_policy(args.benchmark_episodes, opponents=benchmark)
            metrics.log_evaluation(ep, "benchmark", stats)
            if stats["avg_points"] > best_points:
                best_points = stats["avg_points"]
                save_checkpoint(
                    os.path.join(ckpt_dir, f"ppo_{run_id}_best.pt"),
                    policy,
                    build_meta({"benchmark": stats, "benchmark_league": BENCHMARK_LEAGUE, "episode": ep}),
                )
                logger.info(
                    "New best policy at ep=%s | avg_points=%.3f win_rate=%.1f%%",
                    ep, stats["avg_points"], stats["win_rate"] * 100.0,
                )

    try:
        loop.train(num_episodes=args.episodes, on_episode_end=on_episode_end)
        save_policy("final")
        save_checkpoint(os.path.join(ckpt_dir, "ppo_latest.pt"), policy, build_meta())
    finally:
        if gui is not None:
            gui.close()


if __name__ == "__main__":
    main()
