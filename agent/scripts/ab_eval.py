"""A/B evaluation with production visibility and an explicit significance test.

Written after a day in which three separate "improvements" evaporated on closer
inspection:

  * reward shaping "gains" that were inside a +-5 point benchmark spread
  * a lookahead veto measured at +10 points, which came from letting the search
    read the full board when production only sees the fog-of-war view
  * the same veto then measuring -13 and +15 on two fog-limited seeds, i.e. no
    resolvable effect at n=100

The failure mode each time was reading a difference smaller than the noise.
Two agents in one 4-player game interact, so their outcomes are correlated and
the spread is far wider than the binomial +-4pp that n=100 suggests.

So this harness:
  * builds every agent's view through the *fog-limited* payload by default,
    exactly what the /move endpoint receives
  * runs A and B in the same games (paired) and shuffles seats each episode
  * repeats across independent seeds
  * reports a bootstrap confidence interval on the paired difference and says
    plainly whether the result is significant

Usage:
  python agent/scripts/ab_eval.py \\
      --a "veto:best_checkpoint/ppo_league_best.pt:70" \\
      --b "model:best_checkpoint/ppo_league_best.pt" \\
      --opponents baseline:hungry_baseline bot:aggressive_hunter \\
      --episodes 60 --seeds 5
"""

from __future__ import annotations

import argparse
import os
import random
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch

from battlesnake_ai.env.builder import make_env
from battlesnake_ai.inference import search, tactics
from battlesnake_ai.inference.agent_loader import load_agent
from battlesnake_ai.training.action_selection import masked_argmax
from battlesnake_ai.training.baseline_opponents import make_baseline
from battlesnake_ai.training.heuristic_opponents import get_bot_by_name

ACTION_NAMES = ("up", "right", "down", "left")
VIEW_RADIUS = 5


def fog_payload(st_, pid: int, w: int, h: int, radius: int = VIEW_RADIUS) -> Dict[str, Any]:
    """Exactly what the server receives: only what is inside the view radius.

    Opponents outside it appear as fogged entries with null health/length and
    no body, which is how the live API reports them.
    """
    me_body = list(st_.snake_pos.get(pid) or [])
    if not me_body:
        return {"turn": int(st_.turn), "board": {"width": w, "height": h, "food": [],
                "hazards": [], "snakes": []}, "you": {"id": f"s{pid}"}}
    hx, hy = int(me_body[0][0]), int(me_body[0][1])

    def vis(c) -> bool:
        return abs(int(c[0]) - hx) + abs(int(c[1]) - hy) <= radius

    snakes: List[Dict[str, Any]] = []
    for i, alive in enumerate(st_.snakes_alive):
        if not alive:
            continue
        pos = list(st_.snake_pos.get(i) or [])
        if not pos:
            continue
        if i == pid:
            b = [{"x": int(x), "y": int(y)} for x, y in pos]
            snakes.append({"id": f"s{i}", "health": int(st_.snake_health[i]), "body": b,
                           "head": b[0], "length": int(st_.snake_len[i])})
            continue
        if not any(vis(c) for c in pos):
            snakes.append({"id": f"s{i}", "health": None, "length": None, "body": [], "head": None})
            continue
        b = [{"x": int(x), "y": int(y)} for x, y in pos if vis((x, y))]
        if not b:
            snakes.append({"id": f"s{i}", "health": None, "length": None, "body": [], "head": None})
            continue
        snakes.append({"id": f"s{i}", "health": int(st_.snake_health[i]), "body": b,
                       "head": b[0], "length": int(st_.snake_len[i])})

    you = next(s for s in snakes if s["id"] == f"s{pid}")
    food = [{"x": int(f[0]), "y": int(f[1])} for f in st_.food_pos if vis(f)]
    return {"game": {"id": "ab"}, "turn": int(st_.turn),
            "board": {"width": w, "height": h, "food": food, "hazards": [], "snakes": snakes},
            "you": you}


class Agent:
    """One contestant. ``spec`` forms:

      model:<ckpt>            policy + one-step head-to-head filter (production)
      veto:<ckpt>:<budget_ms> policy + lookahead veto
      tactics                 JSON flood-fill/food search
      bot:<name>              omniscient heuristic bot
      baseline:<name>         fog-limited official starter baseline
    """

    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.kind = spec.split(":", 1)[0]
        self.model = None
        self.budget = 70.0
        self.impl = None
        if self.kind in ("model", "veto", "model_space"):
            parts = spec.split(":")
            self.model, _ = load_agent(parts[1], device=torch.device("cpu"))
            self.model.eval()
            if self.kind == "veto" and len(parts) > 2:
                self.budget = float(parts[2])
        elif self.kind == "bot":
            self.impl = get_bot_by_name(spec.split(":", 1)[1])
        elif self.kind == "baseline":
            self.impl = make_baseline(spec.split(":", 1)[1])

    def reset(self) -> None:
        if self.impl is not None and hasattr(self.impl, "reset"):
            self.impl.reset()

    def act(self, env, obs, row: int, pid: int, state, rng: random.Random) -> int:
        legal = list(env.available_actions(pid))
        if self.kind in ("bot", "baseline"):
            a = int(self.impl.select_action(env, pid))
            return a if (not legal or a in legal) else rng.choice(legal)

        w, h = env.cfg.w, env.cfg.h
        payload = fog_payload(state, pid, w, h)

        if self.kind == "tactics":
            mv, _ = tactics.choose_move(payload)
        else:
            with torch.no_grad():
                scores = self.model(obs[row:row + 1]).detach().cpu().numpy()[0]
            order = sorted(range(4), key=lambda i: -scores[i])
            safe = tactics.safe_moves(payload, require_space=(self.kind == "model_space"))
            ranked = [ACTION_NAMES[i] for i in order if ACTION_NAMES[i] in safe] or \
                     [ACTION_NAMES[i] for i in order]
            if self.kind == "veto":
                mv, _ = search.veto(payload, ranked, time_budget_ms=self.budget)
                mv = mv or ranked[0]
            else:
                mv = ranked[0]

        a = ACTION_NAMES.index(mv) if mv else (rng.choice(legal) if legal else 0)
        return a if (not legal or a in legal) else rng.choice(legal)


def run_block(agents: Sequence[Agent], episodes: int, seed: int) -> List[Tuple[float, float]]:
    """Play ``episodes`` games; return per-episode (points_A, points_B)."""
    env = make_env("restricted_standard", num_players=4)
    rng = random.Random(seed)
    out: List[Tuple[float, float]] = []
    for _ in range(episodes):
        seats = list(range(4))
        rng.shuffle(seats)
        for ag in agents:
            ag.reset()
        env.reset()
        done = False
        while not done:
            obs, _, _ = env.get_obs()
            pat = list(env.players_at_turn())
            state = env.get_state()
            acts = []
            for row, pid in enumerate(pat):
                acts.append(agents[seats[pid]].act(env, obs, row, pid, state, rng))
            joint = tuple(acts)
            legal = [tuple(x) for x in env.available_joint_actions()]
            if joint not in legal:
                joint = tuple(rng.choice(legal))
            _, done, _ = env.step(joint)

        s = env.get_state()
        res = []
        for p in range(4):
            surv = (s.turn + 1 if s.snakes_alive[p]
                    else (s.elimination_events[p].turn if p in s.elimination_events else s.turn))
            # Random third key: sorting on the seat index would break exact ties
            # (same survival turn and length) the same way every time, which is
            # an arbitrary thumb on the scale rather than a real placement.
            res.append((surv, int(s.snake_len[p]), rng.random(), p))
        res.sort(reverse=True)
        pts = {}
        for rank, (_, _, _, p) in enumerate(res):
            pts[seats[p]] = [2.0, 1.0, 0.0, 0.0][rank]
        out.append((pts.get(0, 0.0), pts.get(1, 0.0)))
    return out


def bootstrap_ci(diffs: Sequence[float], iters: int = 20000, alpha: float = 0.05
                 ) -> Tuple[float, float]:
    rng = random.Random(12345)
    n = len(diffs)
    means = []
    for _ in range(iters):
        means.append(sum(rng.choice(diffs) for _ in range(n)) / n)
    means.sort()
    return means[int(iters * alpha / 2)], means[int(iters * (1 - alpha / 2))]


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired A/B with production visibility")
    ap.add_argument("--a", required=True, help="challenger spec")
    ap.add_argument("--b", required=True, help="incumbent spec")
    ap.add_argument("--opponents", nargs=2, required=True, help="the other two seats")
    ap.add_argument("--episodes", type=int, default=60, help="episodes per seed")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Shift the seed sequence to get an independent sample")
    args = ap.parse_args()

    agents = [Agent(args.a), Agent(args.b), Agent(args.opponents[0]), Agent(args.opponents[1])]
    print(f"A = {args.a}\nB = {args.b}\nopponents = {args.opponents}")
    print(f"{args.seeds} seeds x {args.episodes} episodes, fog-limited\n")

    all_diffs: List[float] = []
    a_pts: List[float] = []
    b_pts: List[float] = []
    for k in range(args.seeds):
        seed = 1000 + args.seed_offset + k * 137
        block = run_block(agents, args.episodes, seed)
        da = [x[0] for x in block]
        db = [x[1] for x in block]
        all_diffs += [x - y for x, y in zip(da, db)]
        a_pts += da
        b_pts += db
        print(f"  seed {seed}:  A {st.mean(da):.3f} pts   B {st.mean(db):.3f} pts   "
              f"delta {st.mean(da) - st.mean(db):+.3f}")

    n = len(all_diffs)
    mean_diff = st.mean(all_diffs)
    lo, hi = bootstrap_ci(all_diffs)
    print(f"\n  {n} paired games")
    print(f"    A mean points {st.mean(a_pts):.3f}")
    print(f"    B mean points {st.mean(b_pts):.3f}")
    print(f"    difference    {mean_diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    if lo > 0:
        print(f"\n  VERDICT: A is better (CI excludes zero).")
    elif hi < 0:
        print(f"\n  VERDICT: B is better (CI excludes zero).")
    else:
        print(f"\n  VERDICT: no detectable difference. The CI spans zero, so any")
        print(f"           apparent gain here is noise. Do not ship on this.")


if __name__ == "__main__":
    main()
