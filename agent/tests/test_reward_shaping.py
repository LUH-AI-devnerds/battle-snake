"""Reward shaping must distinguish *how* a snake died and who won a collision.

Before this, every death returned the bare env reward, so losing a
head-to-head carried the same signal as starving -- and head-to-head was the
leading cause of death for the trained policy.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.training.ppo_loop import PPOMetricsLogger, PPOTrainingLoop


class _StubEnv:
    num_players = 4
    cfg = SimpleNamespace(w=15, h=15)


def _loop(**kw):
    return PPOTrainingLoop(
        env=_StubEnv(),
        policy=PPOPolicy(in_channels=17),
        metrics=PPOMetricsLogger(logging.getLogger("test")),
        survival_shaping=True,
        **kw,
    )


def _state(alive, *, elim=None, lens=None, health=None, pos=None):
    n = len(alive)
    return SimpleNamespace(
        snakes_alive=list(alive),
        snake_len=list(lens or [5] * n),
        snake_health=list(health or [80] * n),
        snake_pos=dict(pos or {i: [(1 + i, 1)] for i in range(n)}),
        elimination_events=dict(elim or {}),
    )


def test_head_collision_death_is_penalised_more_than_starvation():
    loop = _loop()
    before = _state([True, True])
    h2h = _state([False, True], elim={0: SimpleNamespace(cause="head-collision", turn=9, by="snake-1")})
    starve = _state([False, True], elim={0: SimpleNamespace(cause="out-of-health", turn=9, by=None)})

    r_h2h = loop._reshape_reward(0, -1.0, st_before=before, st_after=h2h, died=True)
    r_starve = loop._reshape_reward(0, -1.0, st_before=before, st_after=starve, died=True)

    assert r_h2h < r_starve, (r_h2h, r_starve)


def test_winning_a_head_to_head_beats_an_unattributed_kill():
    loop = _loop()
    before = _state([True, True, True])
    won = _state(
        [True, False, True],
        elim={1: SimpleNamespace(cause="head-collision", turn=9, by="snake-0")},
    )
    unrelated = _state(
        [True, False, True],
        elim={1: SimpleNamespace(cause="wall-collision", turn=9, by=None)},
        pos={0: [(1, 1)], 1: [(14, 14)], 2: [(7, 7)]},
    )

    r_won = loop._reshape_reward(0, 0.0, st_before=before, st_after=won, died=False)
    r_unrelated = loop._reshape_reward(0, 0.0, st_before=before, st_after=unrelated, died=False)

    assert r_won > r_unrelated, (r_won, r_unrelated)


def test_killed_by_parses_engine_snake_names():
    loop = _loop()
    st = _state([True, False], elim={1: SimpleNamespace(cause="head-collision", turn=3, by="snake-0")})
    assert loop._killed_by(st, 1) == 0
    assert loop._death_cause(st, 1) == "head-collision"
    assert loop._killed_by(st, 0) is None
    assert loop._death_cause(st, 0) is None


def test_shaping_off_returns_base_reward_unchanged():
    loop = PPOTrainingLoop(
        env=_StubEnv(),
        policy=PPOPolicy(in_channels=17),
        metrics=PPOMetricsLogger(logging.getLogger("test")),
        survival_shaping=False,
    )
    before = _state([True, True])
    after = _state([False, True], elim={0: SimpleNamespace(cause="head-collision", turn=9, by="snake-1")})
    assert loop._reshape_reward(0, -1.0, st_before=before, st_after=after, died=True) == -1.0
