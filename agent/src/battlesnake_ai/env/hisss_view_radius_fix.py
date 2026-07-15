"""
Patch hisss ``BattleSnakeGame.get_obs`` view-radius block.

``result`` axis 0 follows ``players_at_turn()`` order (alive snakes only), but
stock hisss indexes rows by raw player id. After eliminations that becomes an
IndexError (e.g. players ``[1, 3]`` with ``result.shape[0] == 2``), and our
server catches it and returns ``FALLBACK_MOVE=up`` every turn.

File mutation works locally but fails silently on read-only containers
(Railway). Prefer rebinding ``get_obs`` from a patched source string in memory;
also try writing the file when the filesystem allows it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MARK_BEGIN = "# --- battlesnake_ai: view_radius row_idx fix begin ---"
_MARK_END = "# --- battlesnake_ai: view_radius row_idx fix end ---"
_NEEDLE = "            for p_self in self.players_at_turn():"
_END_NEEDLE = "            # make mask layer"
_ATTR = "_bs_ai_view_radius_row_fix"


def _fixed_loop_source() -> str:
    return """
            for row_idx, p_self in enumerate(self.players_at_turn()):
                scaled_distance = result[
                    row_idx, :, :, self.layer_explanation["distance_map"]
                ]
                distance_map = scaled_distance * (self.cfg.w + self.cfg.h - 2)
                cur_mask = (distance_map <= self.cfg.view_radius).astype(float)
                masks.append(cur_mask)
                if "current_food" in self.layer_explanation:
                    cur_layer = result[
                        row_idx, :, :, self.layer_explanation["current_food"]
                    ]
                    result[row_idx, :, :, self.layer_explanation["current_food"]] = (
                        cur_layer * cur_mask
                    )
                for p in range(
                    1, self.num_players
                ):  # do not restrict view on own player
                    if f"{p}_snake_health" in self.layer_explanation:
                        result[
                            row_idx, :, :, self.layer_explanation[f"{p}_snake_health"]
                        ] = 0
                    if f"{p}_snake_length" in self.layer_explanation:
                        result[
                            row_idx, :, :, self.layer_explanation[f"{p}_snake_length"]
                        ] = 0
                    if f"{p}_snake_tail_distance" in self.layer_explanation:
                        result[
                            row_idx,
                            :,
                            :,
                            self.layer_explanation[f"{p}_snake_tail_distance"],
                        ] = 0
                    if f"{p}_snake_body" in self.layer_explanation:
                        cur_layer = result[
                            row_idx, :, :, self.layer_explanation[f"{p}_snake_body"]
                        ]
                        result[
                            row_idx, :, :, self.layer_explanation[f"{p}_snake_body"]
                        ] = cur_layer * cur_mask
                    if f"{p}_snake_body_as_one_hot" in self.layer_explanation:
                        cur_layer = result[
                            row_idx,
                            :,
                            :,
                            self.layer_explanation[f"{p}_snake_body_as_one_hot"],
                        ]
                        result[
                            row_idx,
                            :,
                            :,
                            self.layer_explanation[f"{p}_snake_body_as_one_hot"],
                        ] = cur_layer * cur_mask
                    if f"{p}_snake_head" in self.layer_explanation:
                        cur_layer = result[
                            row_idx, :, :, self.layer_explanation[f"{p}_snake_head"]
                        ]
                        result[
                            row_idx, :, :, self.layer_explanation[f"{p}_snake_head"]
                        ] = cur_layer * cur_mask
                    if f"{p}_snake_tail" in self.layer_explanation:
                        cur_layer = result[
                            row_idx, :, :, self.layer_explanation[f"{p}_snake_tail"]
                        ]
                        result[
                            row_idx, :, :, self.layer_explanation[f"{p}_snake_tail"]
                        ] = cur_layer * cur_mask
"""


def _patch_source(text: str) -> Optional[str]:
    if _MARK_BEGIN in text:
        return text
    if "for row_idx, p_self in enumerate(self.players_at_turn()):" in text:
        return text  # already fixed upstream
    i0 = text.find(_NEEDLE)
    i1 = text.find(_END_NEEDLE, i0) if i0 >= 0 else -1
    if i0 < 0 or i1 < 0:
        return None
    return text[:i0] + _MARK_BEGIN + _fixed_loop_source() + "\n" + _MARK_END + "\n" + text[i1:]


def _extract_get_obs_source(module_text: str) -> str:
    lines = module_text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith("    def get_obs("))
    end = next(
        j
        for j in range(start + 1, len(lines))
        if lines[j].startswith("    def ")
    )
    # Dedent one class level so `exec` defines a free function.
    body = lines[start:end]
    return "".join(line[4:] if line.startswith("    ") else line for line in body)


def _rebind_get_obs(module, patched_text: str) -> None:
    ns = dict(vars(module))
    exec(compile(_extract_get_obs_source(patched_text), "<hisss_get_obs_fixed>", "exec"), ns)
    module.BattleSnakeGame.get_obs = ns["get_obs"]
    setattr(module.BattleSnakeGame, _ATTR, True)


def apply_view_radius_row_index_fix() -> bool:
    """
    Ensure ``get_obs`` uses row indices after eliminations.

    Returns True if the fix is active (already applied, file patched, or rebound).
    """
    try:
        import hisss.game.battlesnake as bsm
    except ImportError:
        return False

    if getattr(bsm.BattleSnakeGame, _ATTR, False):
        return True

    path = Path(bsm.__file__).resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Cannot read hisss battlesnake.py for view-radius fix")
        return False

    patched = _patch_source(text)
    if patched is None:
        logger.warning(
            "hisss view-radius patch needle not found; /move may fall back after eliminations"
        )
        return False

    if patched != text:
        try:
            path.write_text(patched, encoding="utf-8")
        except OSError:
            logger.info(
                "hisss site-packages not writable; applying view-radius fix in memory"
            )

    try:
        _rebind_get_obs(bsm, patched)
    except Exception:
        logger.exception("Failed to rebind patched hisss.get_obs")
        return False

    return True
