#!/usr/bin/env python3
"""Watch Battlesnake Blackout leaderboard replays in the local matplotlib GUI."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.blackout.client import fetch_featured_game_id, fetch_leaderboard
from battlesnake_ai.viz.blackout_replay import watch_blackout_game, watch_blackout_leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Battlesnake Blackout games from the online leaderboard"
    )
    parser.add_argument("--game", type=int, default=None, help="Numeric game id (from Watch Game links)")
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Cycle through recent games from top leaderboard snakes",
    )
    parser.add_argument("--featured", action="store_true", help="Play the featured replay on the home page")
    parser.add_argument("--fps", type=float, default=4.0, help="Playback speed (frames per second)")
    parser.add_argument("--list", action="store_true", help="Print leaderboard and exit")
    args = parser.parse_args()

    if args.list:
        for i, e in enumerate(fetch_leaderboard(), start=1):
            rating = f"{e.rating:.2f}" if e.rating is not None else "Unranked"
            print(f"{i:2}. {e.name} (snake/{e.snake_id}) — {rating}")
        return

    if args.game is not None:
        watch_blackout_game(args.game, fps=args.fps)
        return

    if args.featured:
        gid = fetch_featured_game_id()
        if gid is None:
            raise SystemExit("No featured replay on the home page.")
        watch_blackout_game(gid, fps=args.fps, label="featured")
        return

    if args.leaderboard or (args.game is None and not args.featured):
        watch_blackout_leaderboard(fps=args.fps)
        return


if __name__ == "__main__":
    main()
