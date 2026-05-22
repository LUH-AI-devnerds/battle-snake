"""
Matplotlib board view plus a training HUD (metrics, metadata, recent lines).

Coordinates follow hisss state: snake_pos[player] lists (row, col) with head first.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# hisss: UP=0, RIGHT=1, DOWN=2, LEFT=3
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")


def action_tuple_label(actions: Tuple[int, ...]) -> str:
    return "(" + ",".join(ACTION_NAMES[a] if 0 <= a < 4 else str(a) for a in actions) + ")"


def state_to_rgb_board(
    *,
    width: int,
    height: int,
    snake_positions: Dict[int, List[Tuple[int, int]]],
    food_positions,
    alive: Optional[List[bool]] = None,
) -> np.ndarray:
    """
    Build H x W x 3 float image in [0, 1].
    Empty cell: dark gray. Food: gold. Snake A head/body: blues; Snake B: oranges.
    """
    grid = np.zeros((height, width, 3), dtype=np.float32)
    grid[:] = (0.12, 0.12, 0.14)

    for fp in food_positions:
        r, c = int(fp[0]), int(fp[1])
        if 0 <= r < height and 0 <= c < width:
            grid[r, c] = (0.95, 0.75, 0.2)

    colors_head = {
        0: (0.2, 0.55, 1.0),
        1: (1.0, 0.45, 0.15),
        2: (0.35, 0.85, 0.45),
        3: (0.85, 0.35, 0.75),
    }
    colors_body = {
        0: (0.15, 0.35, 0.75),
        1: (0.75, 0.3, 0.1),
        2: (0.2, 0.55, 0.28),
        3: (0.55, 0.2, 0.45),
    }

    for pid, chain in snake_positions.items():
        if alive is not None and pid < len(alive) and not alive[pid]:
            continue
        for i, (r, c) in enumerate(chain):
            r, c = int(r), int(c)
            if not (0 <= r < height and 0 <= c < width):
                continue
            if i == 0:
                grid[r, c] = colors_head.get(pid, (0.5, 0.9, 0.5))
            else:
                grid[r, c] = colors_body.get(pid, (0.3, 0.6, 0.3))

    return grid


def _truncate(s: str, max_len: int = 52) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


class BoardGUI:
    """
    Non-blocking matplotlib UI: board + side panel for training metadata and metrics.
    Call ``set_run_metadata`` once with static fields, then ``update_from_env(..., hud=...)`` each refresh.
    """

    def __init__(
        self,
        title: str = "Battlesnake (hisss)",
        figsize: Tuple[float, float] = (11.0, 6.0),
        recent_log_lines: int = 10,
    ):
        self._title = title
        self._figsize = figsize
        self._fig = None
        self._ax_board = None
        self._ax_hud = None
        self._im = None
        self._hud_text = None
        self._static_meta: Dict[str, Any] = {}
        self._recent: Deque[str] = deque(maxlen=recent_log_lines)

    def set_run_metadata(self, **kwargs: Any) -> None:
        """Merge keys shown in the HUD header (mode, device, hyperparameters, paths, reward_cfg, …)."""
        self._static_meta.update(kwargs)

    def note(self, line: str) -> None:
        """Append a short line to the rolling “Recent” log (e.g. episode end)."""
        self._recent.append(_truncate(line, 70))

    def start(self, width: int, height: int) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        plt.ion()
        self._fig = plt.figure(figsize=self._figsize)
        gs = gridspec.GridSpec(1, 2, figure=self._fig, width_ratios=[1.15, 1.0], wspace=0.28)
        self._ax_board = self._fig.add_subplot(gs[0, 0])
        self._ax_hud = self._fig.add_subplot(gs[0, 1])

        self._ax_board.set_title(self._title)
        self._ax_board.set_xticks([])
        self._ax_board.set_yticks([])
        blank = np.zeros((height, width, 3))
        self._im = self._ax_board.imshow(blank, origin="upper")

        self._ax_hud.set_axis_off()
        self._ax_hud.set_title("Training / logs", loc="left", fontsize=10)
        self._hud_text = self._ax_hud.text(
            0.0,
            1.0,
            "",
            transform=self._ax_hud.transAxes,
            fontsize=8.5,
            family="monospace",
            verticalalignment="top",
            horizontalalignment="left",
            color="0.15",
        )

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def _compose_hud_body(self, hud: Dict[str, Any]) -> str:
        m = self._static_meta
        lines: List[str] = []

        lines.append("══ Run ══")
        if m.get("mode") is not None:
            lines.append(f"  mode={m.get('mode')}   device={m.get('device', '?')}")
        if m.get("obs_shape") is not None:
            lines.append(f"  obs {m.get('obs_shape')}   ch={m.get('in_channels', '?')}")
        if m.get("log_dir"):
            lines.append(f"  log_dir={m.get('log_dir')}")

        lines.append("")
        lines.append("══ Hyperparameters ══")
        lines.append(
            f"  γ={m.get('gamma', hud.get('gamma', '?'))}   lr={m.get('lr', '?')}   "
            f"batch={m.get('batch_size', '?')}"
        )
        lines.append(
            f"  train_after={m.get('train_after', '?')}   "
            f"target_sync={m.get('target_update_every', '?')}   "
            f"ε_decay_steps={m.get('epsilon_decay_steps', '?')}"
        )
        lines.append(f"  replay_cap={m.get('replay_capacity', '?')}")

        rc = m.get("reward_cfg")
        if rc is not None:
            lines.append("")
            lines.append("══ hisss reward_cfg ══")
            lines.append(f"  {rc}")

        lines.append("")
        lines.append("══ Progress ══")
        lines.append(
            f"  episode={hud.get('episode', '?')}   turn={hud.get('turn', '?')}   "
            f"env_steps={hud.get('env_steps', '?')}"
        )
        lines.append(
            f"  optim_steps={hud.get('optim_steps', '?')}   "
            f"ε={_fmt_float(hud.get('epsilon'))}   "
            f"replay={hud.get('replay_fill', '?')}"
        )

        lines.append("")
        lines.append("══ Last env step ══")
        aj = hud.get("action_joint")
        if aj is not None:
            lines.append(f"  actions {action_tuple_label(tuple(aj))}")
        lines.append(f"  step_r={hud.get('step_rewards', '?')}")
        lines.append(f"  ep_return={hud.get('ep_returns', '?')}")
        lines.append(f"  alive={hud.get('snakes_alive', '?')}   health={hud.get('snake_health', '?')}")
        lines.append(f"  done={hud.get('done', '?')}")

        lines.append("")
        lines.append("══ Last DQN update ══")
        if hud.get("had_training"):
            lines.append(
                f"  loss={_fmt_float(hud.get('loss'))}   |TD|={_fmt_float(hud.get('td_abs'))}   "
                f"TD_mean={_fmt_float(hud.get('td_mean'))}"
            )
            lines.append(
                f"  Q(s,a)̄={_fmt_float(hud.get('q_mean'))}   "
                f"max Q_tgt={_fmt_float(hud.get('q_tgt_max'))}   "
                f"∇={_fmt_float(hud.get('grad_norm'))}"
            )
            lines.append(f"  E[r_batch]={_fmt_float(hud.get('r_batch_mean'))}")
            det = hud.get("sample_td_detail")
            if det:
                lines.append(f"  {_truncate(str(det), 54)}")
        else:
            lines.append("  (no grad step yet — collect replay / wait for train_after)")

        if self._recent:
            lines.append("")
            lines.append("══ Recent ══")
            for ln in list(self._recent):
                lines.append(f"  · {ln}")

        return "\n".join(lines)

    def update_from_env(self, env, hud: Optional[Dict[str, Any]] = None) -> None:
        hud = dict(hud or {})
        st = env.get_state()
        cfg = env.cfg
        w, h = int(cfg.w), int(cfg.h)
        if self._fig is None:
            self.start(w, h)

        rgb = state_to_rgb_board(
            width=w,
            height=h,
            snake_positions=dict(st.snake_pos),
            food_positions=st.food_pos,
            alive=list(st.snakes_alive),
        )
        if self._im is not None:
            self._im.set_data(rgb)

        body = self._compose_hud_body(hud)
        if self._hud_text is not None:
            self._hud_text.set_text(body)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def update_from_state(self, env, state, hud: Optional[Dict[str, Any]] = None) -> None:
        hud = dict(hud or {})
        cfg = env.cfg
        w, h = int(cfg.w), int(cfg.h)
        if self._fig is None:
            self.start(w, h)
        rgb = state_to_rgb_board(
            width=w,
            height=h,
            snake_positions=dict(state.snake_pos),
            food_positions=state.food_pos,
            alive=list(state.snakes_alive),
        )
        if self._im is not None:
            self._im.set_data(rgb)
        body = self._compose_hud_body(hud)
        if self._hud_text is not None:
            self._hud_text.set_text(body)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        import matplotlib.pyplot as plt

        if self._fig is not None:
            plt.close(self._fig)
        self._fig = None
        self._ax_board = None
        self._ax_hud = None
        self._im = None
        self._hud_text = None


def _fmt_float(x: Any) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(v) >= 100 or abs(v) < 1e-3:
        return f"{v:.4e}"
    return f"{v:.5f}".rstrip("0").rstrip(".")
