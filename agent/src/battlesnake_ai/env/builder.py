import hisss
from typing import Literal

def make_env(mode: Literal["duel", "standard", "restricted_duel", "restricted_standard"] = "duel") -> hisss.BattleSnakeGame:
    """
    Creates and returns a Battlesnake environment configured for the specific mode.
    """
    if mode == "duel":
        cfg = hisss.duel_config()
    elif mode == "standard":
        cfg = hisss.standard_config()
    elif mode == "restricted_duel":
        cfg = hisss.restricted_duel_config()
    elif mode == "restricted_standard":
        cfg = hisss.restricted_standard_config()
    else:
        raise ValueError(f"Unknown game mode: {mode}")
        
    env = hisss.BattleSnakeGame(cfg)
    return env

def make_custom_duel_env(w: int = 11, h: int = 11) -> hisss.BattleSnakeGame:
    """
    Creates a duel environment with custom width and height.
    """
    cfg = hisss.duel_config()
    cfg.w = w
    cfg.h = h
    return hisss.BattleSnakeGame(cfg)
