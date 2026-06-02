"""
Head-to-head evaluation of trained agents in hisss.

Agent specs:
  random
  dqn:path/to/checkpoint.pt
  rainbow:path/to/checkpoint.pt
  ppo:path/to/checkpoint.pt
  ensemble:rainbow.pt+ppo.pt
  ensemble:rainbow.pt+ppo.pt:0.6:0.4  (optional weights)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.inference.agent_loader import load_agent
from battlesnake_ai.inference.ensemble import EnsembleAgent
from battlesnake_ai.models.dqn import DQN
from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.models.rainbow_dqn import RainbowDQN


@dataclass
class AgentSpec:
    name: str
    kind: str
    path: Optional[str] = None
    ensemble_paths: Optional[Tuple[str, str]] = None
    w_rainbow: float = 0.5
    w_ppo: float = 0.5


@dataclass
class MatchStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_steps: int = 0
    episodes: int = 0


def parse_agent_spec(spec: str) -> AgentSpec:
    if spec == "random":
        return AgentSpec(name="random", kind="random")
    if spec.startswith("ensemble:"):
        body = spec.split(":", 1)[1]
        parts = body.split(":")
        paths = parts[0].split("+")
        if len(paths) != 2:
            raise ValueError(f"Ensemble needs two paths joined by '+': {spec}")
        w_r, w_p = 0.5, 0.5
        if len(parts) >= 3:
            w_r, w_p = float(parts[1]), float(parts[2])
        return AgentSpec(
            name=f"ensemble({os.path.basename(paths[0])}+{os.path.basename(paths[1])})",
            kind="ensemble",
            ensemble_paths=(paths[0], paths[1]),
            w_rainbow=w_r,
            w_ppo=w_p,
        )
    if ":" not in spec:
        raise ValueError(f"Invalid agent spec: {spec}")
    kind, path = spec.split(":", 1)
    if kind not in ("dqn", "rainbow", "ppo", "ppo_finetune"):
        raise ValueError(f"Unknown agent kind: {kind}")
    return AgentSpec(name=f"{kind}({os.path.basename(path)})", kind=kind, path=path)


def load_players(specs: List[AgentSpec], device: torch.device) -> List[Any]:
    players: List[Any] = []
    for s in specs:
        if s.kind == "random":
            players.append(None)
        elif s.kind == "ensemble":
            assert s.ensemble_paths is not None
            players.append(
                EnsembleAgent.from_checkpoints(
                    s.ensemble_paths[0],
                    s.ensemble_paths[1],
                    w_rainbow=s.w_rainbow,
                    w_ppo=s.w_ppo,
                    device=device,
                )
            )
        else:
            model, _ = load_agent(s.path, device=device)
            players.append(model)
    return players


def act_for_player(
    player: Any,
    env: Any,
    obs: np.ndarray,
    row_idx: int,
    pid: int,
) -> int:
    if player is None:
        la = env.available_actions(pid)
        return int(random.choice(la))
    sl = obs[row_idx : row_idx + 1]
    if isinstance(player, EnsembleAgent):
        joint = player.select_joint_actions(env, obs)
        return joint[row_idx]
    if isinstance(player, (DQN, RainbowDQN)):
        with torch.no_grad():
            q = player(sl).detach().cpu().numpy()[0]
        la = env.available_actions(pid)
        best = la[0]
        best_v = q[best]
        for a in la[1:]:
            if q[a] > best_v:
                best_v = q[a]
                best = a
        return int(best)
    if isinstance(player, PPOPolicy):
        with torch.no_grad():
            logits = player.actor_logits(sl)[0]
        la = env.available_actions(pid)
        mask = np.full(logits.shape, -1e9, dtype=np.float32)
        for a in la:
            mask[a] = logits[a]
        return int(mask.argmax())
    raise TypeError(f"Unknown player type: {type(player)}")


def run_episode_two_agents(
    env: Any,
    agent0: Any,
    agent1: Any,
    rng: random.Random,
    *,
    seat0: int = 0,
    seat1: int = 1,
) -> Tuple[int, int]:
    """
    Run one duel. agent0 controls snake ``seat0``, agent1 controls ``seat1``.
    Returns (winner agent index 0|1|-1 draw, env steps).
    """
    env.reset()
    steps = 0
    done = False
    last_rewards: Optional[np.ndarray] = None

    while not done:
        obs, _, _ = env.get_obs()
        pat = list(env.players_at_turn())
        if len(pat) != 2:
            raise RuntimeError("Eval expects exactly 2 snakes per turn")

        actions_list: List[int] = []
        for row_idx, pid in enumerate(pat):
            if pid == seat0:
                actions_list.append(act_for_player(agent0, env, obs, row_idx, pid))
            elif pid == seat1:
                actions_list.append(act_for_player(agent1, env, obs, row_idx, pid))
            else:
                la = env.available_actions(pid)
                actions_list.append(int(rng.choice(la)))

        joint = tuple(actions_list)
        legal = [tuple(x) for x in env.available_joint_actions()]
        if joint not in legal:
            joint = tuple(rng.choice(legal))

        last_rewards, done, _ = env.step(joint)
        steps += 1

    if last_rewards is None:
        return -1, steps
    r0 = float(last_rewards[seat0])
    r1 = float(last_rewards[seat1])
    if r0 > r1:
        return 0, steps
    if r1 > r0:
        return 1, steps
    return -1, steps


def evaluate_pair(
    env: Any,
    agent_a: Any,
    agent_b: Any,
    episodes: int,
    seed: int,
    name_a: str,
    name_b: str,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    stats_a = MatchStats()
    stats_b = MatchStats()

    for ep in range(episodes):
        swap = ep % 2 == 1
        if swap:
            winner, steps = run_episode_two_agents(env, agent_b, agent_a, rng, seat0=0, seat1=1)
            winner_agent = 1 if winner == 0 else (0 if winner == 1 else -1)
        else:
            winner, steps = run_episode_two_agents(env, agent_a, agent_b, rng, seat0=0, seat1=1)
            winner_agent = winner

        if winner_agent == -1:
            stats_a.draws += 1
            stats_b.draws += 1
        elif winner_agent == 0:
            stats_a.wins += 1
            stats_b.losses += 1
        else:
            stats_b.wins += 1
            stats_a.losses += 1
        stats_a.total_steps += steps
        stats_b.total_steps += steps
        stats_a.episodes += 1
        stats_b.episodes += 1

    return {
        name_a: {
            "wins": stats_a.wins,
            "losses": stats_a.losses,
            "draws": stats_a.draws,
            "win_rate": stats_a.wins / max(episodes, 1),
            "mean_steps": stats_a.total_steps / max(episodes, 1),
        },
        name_b: {
            "wins": stats_b.wins,
            "losses": stats_b.losses,
            "draws": stats_b.draws,
            "win_rate": stats_b.wins / max(episodes, 1),
            "mean_steps": stats_b.total_steps / max(episodes, 1),
        },
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Battlesnake agents head-to-head")
    parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        help="Agent specs: random dqn:path.pt rainbow:path ppo:path ensemble:a.pt+b.pt",
    )
    parser.add_argument("--mode", type=str, default="duel")
    parser.add_argument("--num-players", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--round-robin", action="store_true", help="All pairs; default is each vs random")

    args = parser.parse_args()

    specs = [parse_agent_spec(s) for s in args.agents]
    device = torch.device("cpu")
    players = load_players(specs, device)

    env = make_env(mode=args.mode, num_players=args.num_players)
    if args.num_players is None and "duel" in args.mode:
        pass
    env.reset()

    results: Dict[str, Any] = {
        "mode": args.mode,
        "episodes_per_match": args.episodes,
        "seed": args.seed,
        "matches": [],
    }

    if args.round_robin and len(specs) >= 2:
        for i in range(len(specs)):
            for j in range(i + 1, len(specs)):
                match = evaluate_pair(
                    env,
                    players[i],
                    players[j],
                    args.episodes,
                    args.seed + i * 1000 + j,
                    specs[i].name,
                    specs[j].name,
                )
                match["pair"] = [specs[i].name, specs[j].name]
                results["matches"].append(match)
                print(f"\n=== {specs[i].name} vs {specs[j].name} ===")
                for name in (specs[i].name, specs[j].name):
                    s = match[name]
                    print(
                        f"  {name}: wins={s['wins']} losses={s['losses']} draws={s['draws']} "
                        f"win_rate={s['win_rate']:.1%}"
                    )
    else:
        baseline_idx = next((i for i, s in enumerate(specs) if s.kind == "random"), None)
        if baseline_idx is None:
            baseline_idx = 0
            players.insert(0, None)
            specs.insert(0, AgentSpec(name="random", kind="random"))

        for i, spec in enumerate(specs):
            if i == baseline_idx:
                continue
            match = evaluate_pair(
                env,
                players[i],
                players[baseline_idx],
                args.episodes,
                args.seed + i,
                spec.name,
                specs[baseline_idx].name,
            )
            match["pair"] = [spec.name, specs[baseline_idx].name]
            results["matches"].append(match)
            print(f"\n=== {spec.name} vs {specs[baseline_idx].name} ===")
            for name in match["pair"]:
                s = match[name]
                print(
                    f"  {name}: wins={s['wins']} losses={s['losses']} draws={s['draws']} "
                    f"win_rate={s['win_rate']:.1%}"
                )

    os.makedirs(args.log_dir, exist_ok=True)
    out_path = os.path.join(args.log_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
