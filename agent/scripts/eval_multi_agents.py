"""
Multi-snake evaluation of trained agents in hisss.

Supports evaluating up to 4 distinct agents (mixed types: DQN, Rainbow DQN, PPO, Ensemble, Random)
on a standard 15x15 Battlesnake Blackout board with Fog of War (restricted_standard mode).

Example usage:
  python agent/scripts/eval_multi_agents.py \\
    --agents random rainbow:logs/checkpoints/rainbow_latest.pt ppo:logs/checkpoints/ppo_latest.pt random \\
    --episodes 100
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
from battlesnake_ai.training.action_selection import masked_argmax


@dataclass
class AgentSpec:
    name: str
    kind: str
    path: Optional[str] = None
    ensemble_paths: Optional[Tuple[str, str]] = None
    w_rainbow: float = 0.5
    w_ppo: float = 0.5


@dataclass
class AgentStats:
    agent_spec_name: str
    first_places: int = 0
    second_places: int = 0
    third_places: int = 0
    fourth_places: int = 0
    total_score: float = 0.0
    total_survival_turns: int = 0
    total_end_length: int = 0
    episodes_played: int = 0


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
        def rainbow_scores(sub_obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return player.rainbow(sub_obs).detach().cpu().numpy()[0]

        def ppo_scores(sub_obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return player.ppo.actor_logits(sub_obs).detach().cpu().numpy()[0]

        combined = rainbow_scores(sl) * player.w_rainbow + ppo_scores(sl) * player.w_ppo
        la = env.available_actions(pid)
        return masked_argmax(combined, la)

    if isinstance(player, (DQN, RainbowDQN)):
        with torch.no_grad():
            q = player(sl).detach().cpu().numpy()[0]
        la = env.available_actions(pid)
        return masked_argmax(q, la)

    if isinstance(player, PPOPolicy):
        with torch.no_grad():
            logits = player.actor_logits(sl)[0].detach().cpu().numpy()
        la = env.available_actions(pid)
        mask = np.full(logits.shape, -1e9, dtype=np.float32)
        for a in la:
            mask[a] = logits[a]
        return int(mask.argmax())

    raise TypeError(f"Unknown player type: {type(player)}")


def run_multi_agent_episode(
    env: Any,
    players: List[Any],
    rng: random.Random,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Runs one episode of Battlesnake Blackout.
    players is a list of length env.num_players.
    Returns:
      - list of elimination turns for each player (index is player seat)
      - list of final lengths for each player
      - list of total steps/turns taken
    """
    env.reset()
    done = False
    turns = 0

    while not done:
        obs, _, _ = env.get_obs()
        pat = list(env.players_at_turn())

        actions_list: List[int] = []
        for row_idx, pid in enumerate(pat):
            player = players[pid]
            actions_list.append(act_for_player(player, env, obs, row_idx, pid))

        joint = tuple(actions_list)
        legal = [tuple(x) for x in env.available_joint_actions()]
        if joint not in legal:
            joint = tuple(rng.choice(legal))

        _, done, _ = env.step(joint)
        turns += 1

    st = env.get_state()
    
    elimination_turns = [0] * env.num_players
    final_lengths = [st.snake_len[pid] for pid in range(env.num_players)]
    
    for pid in range(env.num_players):
        if st.snakes_alive[pid]:
            elimination_turns[pid] = st.turn + 1
        elif pid in st.elimination_events:
            elimination_turns[pid] = st.elimination_events[pid].turn
        else:
            elimination_turns[pid] = st.turn

    return elimination_turns, final_lengths, turns


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple Battlesnake agents in 4-player games")
    parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        help="List of 4 agent specifications. E.g., random dqn:path.pt rainbow:path.pt ppo:path.pt",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="restricted_standard",
        help="Game mode (restricted_standard is 15x15 standard Blackout)",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="logs", help="Output log directory")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable seat shuffling (agents will always sit in their listed argument positions)",
    )

    args = parser.parse_args()

    if len(args.agents) != 4:
        # If fewer than 4 are specified, we pad with random agents or repeat
        print(f"WARNING: Blackout requires 4 agents. You provided {len(args.agents)}. Padding with 'random'.")
        while len(args.agents) < 4:
            args.agents.append("random")
        if len(args.agents) > 4:
            args.agents = args.agents[:4]

    specs = [parse_agent_spec(s) for s in args.agents]
    device = torch.device("cpu")
    
    print("Loading models...")
    players = load_players(specs, device)
    
    print(f"Creating env mode={args.mode} (4 players)...")
    env = make_env(mode=args.mode, num_players=4)
    env.reset()

    # Create unique agent names to track stats
    # E.g. "Agent 0: rainbow(latest.pt)", "Agent 1: random"
    agent_names = [f"Agent {i}: {spec.name}" for i, spec in enumerate(specs)]
    
    stats = {
        name: AgentStats(agent_spec_name=name) for name in agent_names
    }

    rng = random.Random(args.seed)

    print(f"Running {args.episodes} episodes...")
    
    for ep in range(1, args.episodes + 1):
        # Determine seats
        seats = list(range(4))
        if not args.no_shuffle:
            rng.shuffle(seats)
        
        # seats[p] = which agent index sits in seat p
        # so players_in_episode[p] is the player object for seat p
        players_in_episode = [players[seats[p]] for p in range(4)]
        
        elim_turns, final_lens, turns = run_multi_agent_episode(env, players_in_episode, rng)
        
        # Rank the seats based on (survival_turn, final_length) descending
        # We also keep track of seat index to attribute stats correctly
        seat_results = []
        for pid in range(4):
            seat_results.append({
                "pid": pid,
                "survival": elim_turns[pid],
                "length": final_lens[pid]
            })
            
        # Sort descending by survival turn, then length
        seat_results.sort(key=lambda x: (x["survival"], x["length"]), reverse=True)
        
        # Placements and Scores:
        # 1st place gets 2 points
        # 2nd place gets 1 point
        # 3rd & 4th get 0 points
        scores_by_rank = [2.0, 1.0, 0.0, 0.0]
        
        for rank, res in enumerate(seat_results):
            pid = res["pid"]
            agent_idx = seats[pid]
            agent_name = agent_names[agent_idx]
            
            ast = stats[agent_name]
            ast.episodes_played += 1
            ast.total_survival_turns += res["survival"]
            ast.total_end_length += res["length"]
            ast.total_score += scores_by_rank[rank]
            
            if rank == 0:
                ast.first_places += 1
            elif rank == 1:
                ast.second_places += 1
            elif rank == 2:
                ast.third_places += 1
            else:
                ast.fourth_places += 1

        if ep % max(1, args.episodes // 10) == 0 or ep == args.episodes:
            print(f" Completed {ep}/{args.episodes} episodes...")

    # Display results
    print("\n" + "="*80)
    print(f" BATTLESNAKE BLACKOUT MULTI-AGENT EVALUATION RESULTS (Mode: {args.mode})")
    print(f" Seed: {args.seed} | Total Episodes: {args.episodes}")
    print("="*80)
    
    header = f"{'Agent / Algorithm':<42} | {'1st':<5} {'2nd':<5} {'3rd':<5} {'4th':<5} | {'Win%':<7} | {'Avg Pts':<8} | {'Avg Turn':<8} | {'Avg Len':<7}"
    print(header)
    print("-" * len(header))
    
    json_summary = []
    
    for name in agent_names:
        ast = stats[name]
        ep_played = max(1, ast.episodes_played)
        win_rate = ast.first_places / ep_played
        sec_rate = ast.second_places / ep_played
        avg_score = ast.total_score / ep_played
        avg_survival = ast.total_survival_turns / ep_played
        avg_len = ast.total_end_length / ep_played
        
        row = (
            f"{name:<42} | "
            f"{ast.first_places:<5} {ast.second_places:<5} {ast.third_places:<5} {ast.fourth_places:<5} | "
            f"{win_rate:6.1%} | "
            f"{avg_score:<8.2f} | "
            f"{avg_survival:<8.1f} | "
            f"{avg_len:<7.1f}"
        )
        print(row)
        
        json_summary.append({
            "agent_name": name,
            "1st": ast.first_places,
            "2nd": ast.second_places,
            "3rd": ast.third_places,
            "4th": ast.fourth_places,
            "win_rate": win_rate,
            "avg_score": avg_score,
            "avg_survival": avg_survival,
            "avg_length": avg_len,
        })
        
    print("="*80)
    
    # Save results
    os.makedirs(args.log_dir, exist_ok=True)
    out_path = os.path.join(args.log_dir, f"eval_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "episodes": args.episodes,
            "seed": args.seed,
            "results": json_summary
        }, f, indent=2)
    print(f"Wrote full results to {out_path}\n")


if __name__ == "__main__":
    main()
