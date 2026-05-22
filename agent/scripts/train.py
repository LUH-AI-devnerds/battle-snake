import argparse
import sys
import os

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.models.simple_cnn import SimpleCNN
from battlesnake_ai.training.logger import setup_logger, get_tensorboard_writer
from battlesnake_ai.training.loop import TrainingLoop

def main():
    parser = argparse.ArgumentParser(description="Train a Battlesnake AI model")
    parser.add_argument("--mode", type=str, default="restricted_standard", choices=["duel", "standard", "restricted_duel", "restricted_standard"], help="Game mode")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to simulate")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs")
    args = parser.parse_args()

    # 1. Setup Logging
    logger = setup_logger(log_dir=args.log_dir, log_name="train")
    tb_writer = get_tensorboard_writer(log_dir=os.path.join(args.log_dir, "tensorboard"))

    logger.info(f"Initializing {args.mode} environment...")
    
    # 2. Build Environment
    env = make_env(mode=args.mode)
    
    # 3. Build Model
    env.reset()
    obs, _, _ = env.get_obs()
    in_channels = obs.shape[-1]
    
    logger.info(f"Observation shape detected: {obs.shape}. In channels: {in_channels}")
    model = SimpleCNN(in_channels=in_channels, num_actions=4)
    
    # 4. Run Training
    loop = TrainingLoop(env=env, model=model, logger=logger, tensorboard_writer=tb_writer)
    loop.train(num_episodes=args.episodes)
    
if __name__ == "__main__":
    main()
