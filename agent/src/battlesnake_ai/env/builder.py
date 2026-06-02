import hisss
from typing import Literal, Optional


def _apply_num_players(cfg: hisss.BattleSnakeConfig, num_players: int) -> None:
    """Resize player count and keep per-snake length / spawn metadata consistent."""
    if num_players < 1:
        raise ValueError(f"num_players must be >= 1, got {num_players}")

    default_len = cfg.init_snake_len[0] if cfg.init_snake_len else 3
    cfg.num_players = num_players
    cfg.init_snake_len = [default_len] * num_players

    if cfg.init_snake_pos is not None and len(cfg.init_snake_pos) != num_players:
        cfg.init_snake_pos = None


def make_env(
    mode: Literal["duel", "standard", "restricted_duel", "restricted_standard"] = "restricted_standard",
    *,
    num_players: Optional[int] = None,
) -> hisss.BattleSnakeGame:
    """
    Creates and returns a Battlesnake environment configured for the specific mode.

    ``num_players`` overrides the default snake count for the mode (duel defaults to 2,
    standard-family modes default to 4). Duel / restricted_duel configs are normally two-player;
    use standard modes when training with more than two snakes.
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

    if num_players is not None:
        if mode in ("duel", "restricted_duel") and num_players != 2:
            raise ValueError(
                f"mode {mode!r} is a two-snake duel; use a standard mode or omit --num-players"
            )
        _apply_num_players(cfg, num_players)

    return hisss.BattleSnakeGame(cfg)

def make_custom_duel_env(w: int = 11, h: int = 11) -> hisss.BattleSnakeGame:
    """
    Creates a duel environment with custom width and height.
    """
    cfg = hisss.duel_config()
    cfg.w = w
    cfg.h = h
    return hisss.BattleSnakeGame(cfg)
