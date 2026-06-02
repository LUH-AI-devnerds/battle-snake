"""
Matplotlib playback for Battlesnake Blackout leaderboard replays.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

from battlesnake_ai.blackout.client import fetch_replay, sample_leaderboard_game_ids
from battlesnake_ai.blackout.replay import BlackoutReplay
from battlesnake_ai.viz.board_gui import BoardGUI, state_to_rgb_board


class BlackoutReplayViewer:
    def __init__(self, title: str = "Blackout replay", *, gui: Optional[BoardGUI] = None):
        self.gui = gui if gui is not None else BoardGUI(title=title)
        self._owns_gui = gui is None

    def close(self) -> None:
        if self._owns_gui:
            self.gui.close()

    def show_turn(self, replay: BlackoutReplay, turn: int, *, extra_hud: Optional[Dict[str, Any]] = None) -> None:
        w, h, snake_pos, food, alive = replay.frame(turn)
        if self.gui._fig is None:
            self.gui.start(w, h)

        rgb = state_to_rgb_board(
            width=w,
            height=h,
            snake_positions=snake_pos,
            food_positions=food,
            alive=alive,
        )
        if self.gui._im is not None:
            self.gui._im.set_data(rgb)

        info = replay.snake_info_at(turn)
        labels = [f"P{i}:{replay.snake_labels.get(i, '?')}" for i in range(len(alive))]
        health_by_pid = {row["pid"]: row.get("health") for row in info if row.get("pid") is not None}

        hud: Dict[str, Any] = {
            "episode": f"game {replay.game_id}",
            "turn": turn,
            "env_steps": f"{turn}/{replay.total_turns}",
            "snakes_alive": alive,
            "snake_health": [health_by_pid.get(i, "?") for i in range(len(alive))],
            "done": turn >= len(replay.raw["moves"]) - 1,
        }
        if extra_hud:
            hud.update(extra_hud)

        body_lines = [
            "══ Blackout replay ══",
            f"  game_id={replay.game_id}   turn={turn}/{replay.total_turns}",
            f"  board={w}×{h}   snakes={len(alive)}",
            "",
            "══ Snakes ══",
        ]
        for i, lbl in enumerate(labels):
            hval = health_by_pid.get(i, "?")
            status = "alive" if i < len(alive) and alive[i] else "dead"
            body_lines.append(f"  {lbl}  hp={hval}  ({status})")

        if extra_hud:
            body_lines.append("")
            body_lines.append("══ Notes ══")
            for k, v in extra_hud.items():
                body_lines.append(f"  {k}={v}")

        if self.gui._hud_text is not None:
            self.gui._hud_text.set_text("\n".join(body_lines))

        if self.gui._fig is not None:
            self.gui._fig.canvas.draw_idle()
            self.gui._fig.canvas.flush_events()

    def play(
        self,
        replay: BlackoutReplay,
        *,
        fps: float = 4.0,
        start_turn: int = 0,
        extra_hud: Optional[Dict[str, Any]] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        delay = 1.0 / max(0.25, fps)
        n = len(replay.raw["moves"])
        for turn in range(start_turn, n):
            if not plt.fignum_exists(self.gui._fig.number):
                break
            self.show_turn(replay, turn, extra_hud=extra_hud)
            plt.pause(delay)

    def play_game_id(
        self,
        game_id: int,
        *,
        fps: float = 4.0,
        label: Optional[str] = None,
    ) -> None:
        data = fetch_replay(game_id)
        replay = BlackoutReplay.from_api(data, game_id=game_id)
        title = label or f"Blackout #{game_id}"
        self.gui._title = title
        if self.gui._ax_board is not None:
            self.gui._ax_board.set_title(title)
        extra = {"source": "bs-blackout-2026", "label": label} if label else {"source": "bs-blackout-2026"}
        self.play(replay, fps=fps, extra_hud=extra)

    def play_random_leaderboard(
        self, *, fps: float = 4.0, pool: Optional[List[Tuple[int, str]]] = None
    ) -> Optional[int]:
        pool = pool or sample_leaderboard_game_ids()
        if not pool:
            return None
        game_id, label = random.choice(pool)
        self.play_game_id(int(game_id), fps=fps, label=label)
        return int(game_id)


def watch_blackout_game(
    game_id: int,
    *,
    fps: float = 4.0,
    label: Optional[str] = None,
) -> None:
    viewer = BlackoutReplayViewer()
    try:
        viewer.play_game_id(game_id, fps=fps, label=label)
        import matplotlib.pyplot as plt

        if viewer.gui._fig is not None and plt.fignum_exists(viewer.gui._fig.number):
            plt.ioff()
            plt.show()
    finally:
        viewer.close()


def watch_blackout_leaderboard(*, fps: float = 4.0, shuffle: bool = True) -> None:
    pool = sample_leaderboard_game_ids()
    if shuffle:
        random.shuffle(pool)
    viewer = BlackoutReplayViewer(title="Blackout leaderboard")
    try:
        for game_id, label in pool:
            import matplotlib.pyplot as plt

            if viewer.gui._fig is not None and not plt.fignum_exists(viewer.gui._fig.number):
                break
            viewer.play_game_id(game_id, fps=fps, label=label)
            if viewer.gui._fig is None or not plt.fignum_exists(viewer.gui._fig.number):
                break
            time.sleep(0.5)
        import matplotlib.pyplot as plt

        if viewer.gui._fig is not None and plt.fignum_exists(viewer.gui._fig.number):
            plt.ioff()
            plt.show()
    finally:
        viewer.close()
