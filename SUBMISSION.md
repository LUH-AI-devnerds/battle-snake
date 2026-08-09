# Final submission state

## What is deployed

| | |
|---|---|
| URL | https://web-production-01418.up.railway.app |
| checkpoint | `best_checkpoint/ppo_league_best.pt` (PPO, ep19500) |
| strategy | `MOVE_STRATEGY=model_space` — policy ranks moves; the filter rejects suicide, losing head-to-heads, **and pockets too small to hold us** |
| threads | `torch` pinned to 1 (see server.py) |

## Why this checkpoint

Every alternative was compared against it head-to-head, in the same games,
under fog-of-war visibility, with a bootstrap confidence interval
(`agent/scripts/ab_eval.py`). None was better:

| candidate | result vs deployed |
|---|---|
| `ppo_20260808_154148` (baseline league, ep3250) | −0.025 pts, CI [−0.158, +0.111] — no difference |
| `ppo_20260807_181121` (head-to-head shaping, ep500) | −0.015 pts, CI [−0.190, +0.165] — no difference |
| `rainbow_v2_best` | far weaker in every head-to-head field measured |
| `ppo_bc_teacher` | the champion's own precursor, superseded by 20k PPO episodes |
| `ppo_best` (original) | the broken checkpoint, roughly random strength |

Self-reported training benchmarks are **not** evidence: the ep3250 checkpoint
scored 69.3% on its own benchmark against 62.7% for the starting policy, and
that entire gap disappeared under paired evaluation. The noise floor is about
±0.13 points at 360 paired games.

`MOVE_STRATEGY=veto` (lookahead search) is implemented but **off**. It measured
+10 points only because the harness let the search read the full board; under
real fog-limited visibility it was −13 on one seed and +15 on another, i.e. no
resolvable effect, and it consumed ~40% of the real latency headroom.

## The one change that earned its place

Adding a space check to the move filter is worth **+0.146 points/game**,
95% CI [+0.053, +0.239], over 720 paired fog-limited games across 12
independent seeds (11/12 positive, sign-test p = 0.006). Head-to-head
avoidance alone cannot see a move that is safe this turn and a sealed dead
end four turns later, and self-trapping produces the early eliminations that
cost the most rating.

Everything else tried today failed the same bar and was **not** shipped:
reward shaping, the lookahead veto, and both retrained checkpoints.

## Latency reality

`/health` does no computation and still takes 217–296 ms from outside, so
roughly half of the 500 ms budget is network before our code runs. Our compute
is 9–20 ms (p99 ~21 ms). Do not spend the apparent headroom; it is not there.

## Before changing anything

    ./scripts/preflight.sh --live

7 checks, each one covering something that actually broke in production. All
must pass. Run it on an idle machine — under load it reports meaningless
latency and can even core-dump pytest through resource contention.

## During a run

    ./scripts/watch_live.sh

Watch `fallbacks`. Above zero means the RL policy stopped driving and the crude
heuristic answered — that is a defect, not a slow patch. `/stats` and
`/stats/moves` expose the same data over HTTP.

## Known-good invariants

- `/move`, `/start`, `/end` never return 5xx, whatever the payload
- a null field means "unknown", never "dead" — under fog of war opponents
  report `health: null`, and treating that as 0 made hisss see a sole-survivor
  board and skip the model for entire games
- the runtime lock is required: hisss is not thread-safe and concurrent
  `/move` calls previously corrupted memory and core-dumped the process
- the code default for `MOVE_STRATEGY` must equal the Dockerfile's; a test
  enforces it
