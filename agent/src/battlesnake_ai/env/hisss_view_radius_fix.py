"""
Patch hisss ``BattleSnakeGame.get_obs`` view-radius block: ``result`` axis 0 follows
``players_at_turn()`` order, but hisss indexed rows by raw player id (breaks when ids
are non-contiguous after eliminations). Idempotent file patch on ``hisss.game.battlesnake``.
"""

from __future__ import annotations

from pathlib import Path


_MARK_BEGIN = "# --- battlesnake_ai: view_radius row_idx fix begin ---"
_MARK_END = "# --- battlesnake_ai: view_radius row_idx fix end ---"


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
                    food_idx = self.layer_explanation["current_food"]
                    cur_layer = result[row_idx, :, :, food_idx]
                    result[row_idx, :, :, food_idx] = cur_layer * cur_mask
                    # Food that spawned this turn is always visible for one step
                    if new_food_pos is not None and len(new_food_pos) > 0:
                        spawn_mask = self._new_food_obs_mask(
                            new_food_pos, p_self, num_rot, flip
                        )
                        result[row_idx, :, :, food_idx] = np.maximum(
                            result[row_idx, :, :, food_idx], spawn_mask
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


def apply_view_radius_row_index_fix() -> bool:
    """
    Patch installed ``hisss`` once if needed. Returns True if file was modified or already patched.
    """
    try:
        import hisss.game.battlesnake as bsm
    except ImportError:
        return False

    path = Path(bsm.__file__).resolve()
    text = path.read_text(encoding="utf-8")
    if _MARK_BEGIN in text:
        return True

    needle = "            for p_self in self.players_at_turn():"
    if needle not in text:
        return False

    # Original buggy block ends before "            # make mask layer"
    end_needle = "            # make mask layer"
    i0 = text.find(needle)
    i1 = text.find(end_needle, i0)
    if i0 < 0 or i1 < 0:
        return False

    replacement = _MARK_BEGIN + _fixed_loop_source() + "\n" + _MARK_END + "\n"
    new_text = text[:i0] + replacement + text[i1:]
    path.write_text(new_text, encoding="utf-8")

    import importlib

    importlib.reload(bsm)
    return True
