import hisss
from hisss import UP, DOWN, LEFT, RIGHT
import numpy as np
import torch
import torch.nn as nn
   

def create_example_board_env():
    """
    Creates an example Battlesnake game with a custom starting board layout.
    """
    # Create a base configuration (duel)
    game_config = hisss.duel_config()
    
    # Customize the board size
    game_config.w = 7
    game_config.h = 7
    
    # Customize the snake starting positions
    # Snake 0: Head at [1, 1], body at [1, 2], tail at [1, 3]
    # Snake 1: Head at [5, 5], body at [5, 4], tail at [5, 3]
    game_config.init_snake_pos = {
        0: [[1, 1], [1, 2], [1, 3]],
        1: [[5, 5], [5, 4], [5, 3]]
    }
    
    # Initial food positions
    game_config.init_food_pos = [[3, 3], [1, 5], [5, 1]]
    
    # Set initial lengths to match the provided positions
    game_config.init_snake_len = [3, 3]
    
    # Disable random spawning config mismatches
    game_config.min_food = 1
    
    # Initialize the environment
    env = hisss.BattleSnakeGame(game_config)
    return env

class DummyModel(nn.Module):
    """
    A dummy PyTorch model that takes the Battlesnake observation as input
    and predicts logits for the 4 possible actions.
    """
    def __init__(self, in_channels: int, num_actions: int = 4):
        super().__init__()
        # A simple Convolutional Neural Network
        self.net = nn.Sequential(
            # The observation shape from hisss is typically (N, W, H, C)
            # But PyTorch expects (N, C, W, H). We will permute in forward.
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_actions)
        )

    def forward(self, obs_np: np.ndarray) -> torch.Tensor:
        """
        Args:
            obs_np: numpy array of shape (batch, width, height, channels)
        Returns:
            Logits for each action of shape (batch, 4)
        """
        # Convert to torch tensor
        x = torch.from_numpy(obs_np).float()
            
        # Permute from (B, W, H, C) to (B, C, W, H) for PyTorch Conv2d
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        
        # Forward pass
        logits = self.net(x)
        return logits


def main():
    print("Initializing example board...")
    env = create_example_board_env()
    env.reset()
    
    # Render the initial board
    print("Initial Board:")
    env.render()
    
    # Get initial observation
    # get_obs() typically returns (obs, perm, inv_perm)
    obs, _, _ = env.get_obs()
    
    print(f"Observation shape: {obs.shape}")
    # obs shape is typically (num_snakes_alive, w, h, channels)
    in_channels = obs.shape[-1]
    print("input channels")
    print(in_channels)
    
    print("Initializing dummy model...")
    model = DummyModel(in_channels=in_channels, num_actions=4)
    
    print("\n--- Starting Game Loop ---")
    done = False
    turn = 0
    while not done and turn < 10:  # Play max 10 turns for example
        turn += 1
        print(f"\nTurn {turn}:")
        
        # 1. Get Observation
        obs, _, _ = env.get_obs()
        
        # 2. Forward pass through Dummy Model to get action logits
        logits = model(obs)
        # Pick the action with the highest logit for each snake
        actions = torch.argmax(logits, dim=-1).tolist()
            
        print(f"Model selected actions: {actions}")
        
        # Verify action length matches players at turn
        # The environment expects actions only for the players currently at turn
        # If a snake died, it won't be in players_at_turn()
        players_at_turn = env.players_at_turn()
        valid_actions = tuple(actions[i] for i in range(len(players_at_turn)))
        
        # 3. Step the environment
        try:
            rewards, done, _ = env.step(actions=valid_actions)
        except ValueError as e:
            # E.g., if we tried an illegal joint action (like moving into death and all_actions_legal=False)
            print(f"Encountered ValueError: {e}")
            print("Choosing random available actions instead.")
            # Fallback to random legal actions
            valid_actions = env.available_joint_actions()[0] 
            rewards, done, _ = env.step(actions=valid_actions)
            
        env.render()
        print(f"Rewards: {rewards}")
        
    print("\nGame Over or Reached Turn Limit!")
    env.close()

if __name__ == "__main__":
    main()
