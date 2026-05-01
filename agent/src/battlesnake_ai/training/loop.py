import torch
import numpy as np
from typing import Dict, Any

from battlesnake_ai.models.base import BaseModel
import hisss

class TrainingLoop:
    def __init__(self, env: hisss.BattleSnakeGame, model: BaseModel, logger, tensorboard_writer=None):
        self.env = env
        self.model = model
        self.logger = logger
        self.tb = tensorboard_writer
        self.global_step = 0

    def run_episode(self) -> Dict[str, Any]:
        """
        Runs a single episode of the game.
        For now, this assumes a simple self-play or solo-play depending on the env config,
        where the model acts for all alive snakes.
        """
        self.env.reset()
        done = False
        total_rewards = np.zeros(self.env.num_players)
        turns = 0

        while not done:
            obs, _, _ = self.env.get_obs()
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(obs)
                # Greedy action selection for demonstration
                actions = torch.argmax(logits, dim=-1).tolist()
                
            players_at_turn = self.env.players_at_turn()
            valid_actions = tuple(actions[i] for i in range(len(players_at_turn)))
            
            try:
                rewards, done, _ = self.env.step(valid_actions)
                total_rewards += rewards
                turns += 1
            except ValueError as e:
                self.logger.warning(f"Illegal action taken: {valid_actions}. Fallback to random.")
                valid_actions = self.env.available_joint_actions()[0]
                rewards, done, _ = self.env.step(valid_actions)
                total_rewards += rewards
                turns += 1

        stats = {
            "turns": turns,
            "rewards": total_rewards,
        }
        return stats

    def train(self, num_episodes: int):
        self.logger.info(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(1, num_episodes + 1):
            stats = self.run_episode()
            self.global_step += 1
            
            # Log to console
            self.logger.info(f"Episode {episode} | Turns: {stats['turns']} | Rewards: {stats['rewards']}")
            
            # Log to TensorBoard
            if self.tb is not None:
                self.tb.add_scalar("Train/Turns", stats['turns'], self.global_step)
                for p in range(self.env.num_players):
                    self.tb.add_scalar(f"Train/Reward_Player_{p}", stats['rewards'][p], self.global_step)
                    
        self.logger.info("Training completed.")
        if self.tb:
            self.tb.close()
