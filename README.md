# Battle Snake

Train reinforcement-learning agents for [Battlesnake](https://play.battlesnake.com/) using the [hisss](https://github.com/BattlesnakeOfficial/hisss) simulator. This repo includes a PyTorch training stack (CNN baseline, DQN, Rainbow DQN, PPO), checkpoint I/O, head-to-head evaluation, optional live board visualization, TensorBoard metrics, and a minimal FastAPI server stub for deploying a snake to the official API.

## Project layout

```
battle-snake/
├── agent/                    # Training package and scripts
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── train.py              # Greedy CNN rollout (no gradients)
│   │   ├── train_dqn.py          # Vanilla DQN
│   │   ├── train_rainbow.py      # Rainbow DQN (PER + distributional)
│   │   ├── train_ppo.py          # PPO actor-critic
│   │   ├── train_ppo_from_rainbow.py  # Sequential hybrid fine-tune
│   │   └── eval_agents.py        # Head-to-head eval vs random / each other
│   └── src/battlesnake_ai/
│       ├── env/              # hisss env factory + view-radius patch
│       ├── models/           # Backbone, DQN, Rainbow, PPO
│       ├── training/         # Loops, replay, checkpoints, action selection
│       ├── inference/        # Checkpoint loader, ensemble policy
│       └── viz/              # Matplotlib training GUI
├── example_board_and_model.py  # Standalone hisss + dummy CNN demo
├── server.py                 # Blackout 2026 FastAPI server
├── Dockerfile / Procfile     # Deployment
└── scripts/test_blackout_api.py
```

## Requirements

- Python 3.10+ (recommended)
- [hisss](https://github.com/BattlesnakeOfficial/hisss) — Battlesnake game engine and observation tensors
- PyTorch, NumPy, TensorBoard, Matplotlib (see `agent/requirements.txt`)

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r agent/requirements.txt
```

Install **hisss** if it is not already available (for example from the official repo or a local clone). The `.gitignore` excludes a top-level `hisss/` directory when you vendor the simulator locally.

Run training scripts from the `agent/` directory so logs land in `agent/logs/` by default:

```bash
cd agent
```

## Quick start

**Custom board + random policy steps** (no training package):

```bash
python example_board_and_model.py
```

**Greedy CNN episodes** (forward pass only, logs turns/rewards):

```bash
python scripts/train.py --mode restricted_standard --episodes 10
```

**DQN training** (replay buffer, target network, ε-decay):

```bash
python scripts/train_dqn.py --mode restricted_standard --episodes 50
```

**DQN with live board GUI:**

```bash
python scripts/train_dqn.py --mode duel --episodes 20 --gui --gui-every 2
```

**Rainbow DQN** (dueling + C51 + prioritized replay):

```bash
python scripts/train_rainbow.py --mode duel --episodes 500 --checkpoint-every 50
```

**Rainbow DQN with live board GUI:**

```bash
python scripts/train_rainbow.py --mode duel --episodes 20 --gui --gui-every 2
```

**PPO**:

```bash
python scripts/train_ppo.py --mode duel --episodes 200 --rollout-steps 2048
```

**Sequential hybrid** (Rainbow encoder → PPO fine-tune):

```bash
python scripts/train_ppo_from_rainbow.py \
  --rainbow-checkpoint logs/checkpoints/rainbow_latest.pt \
  --mode duel --episodes 300
```

**Compare agents** (checkpoints vs random or round-robin):

```bash
python scripts/eval_agents.py --mode duel --episodes 200 \
  --agents random dqn:logs/checkpoints/dqn_latest.pt \
  rainbow:logs/checkpoints/rainbow_latest.pt \
  ppo:logs/checkpoints/ppo_latest.pt \
  ensemble:logs/checkpoints/rainbow_latest.pt+logs/checkpoints/ppo_latest.pt

python scripts/eval_agents.py --round-robin --episodes 100 \
  --agents dqn:logs/checkpoints/dqn_latest.pt rainbow:logs/checkpoints/rainbow_latest.pt
```

**Watch [Battlesnake Blackout](https://www.tnt.uni-hannover.de/bs-blackout-2026/) leaderboard replays locally:**

```bash
# List current leaderboard
python scripts/watch_blackout.py --list

# One game (id from "Watch Game" on the site)
python scripts/watch_blackout.py --game 33946

# Cycle recent games from top snakes
python scripts/watch_blackout.py --leaderboard
```

**DQN training + Blackout replays in the same GUI window:**

```bash
# Watch game 33946 before training, then train with live board
python scripts/train_dqn.py --gui --blackout-replay 33946 --episodes 20

# Every 5 episodes, replay a random recent leaderboard match
python scripts/train_dqn.py --gui --blackout-replay-every 5 --episodes 50
```

## Game modes

`make_env()` supports four hisss configurations:

| Mode | Description |
|------|-------------|
| `duel` | Two snakes |
| `standard` | Standard multi-snake rules |
| `restricted_duel` | Duel with restricted ruleset |
| `restricted_standard` | Default for training scripts |

Pass `--mode` to either training script. The DQN loop supports simultaneous moves and joint legal actions for any player count hisss allows.

### Number of snakes

| Mode family | Default snakes | Override |
|-------------|----------------|----------|
| `duel`, `restricted_duel` | 2 | Fixed at 2 (duel ruleset) |
| `standard`, `restricted_standard` | 4 | `--num-players N` |

Examples:

```bash
# Four snakes (default for restricted_standard)
python scripts/train_dqn.py --mode restricted_standard --episodes 50

# Six-snake free-for-all on the standard ruleset
python scripts/train_dqn.py --mode standard --num-players 6 --episodes 50
```

`make_env(mode, num_players=N)` applies the same override in Python. hisss also lets you set `num_players`, board size, and spawn positions on a `BattleSnakeConfig` directly (see `example_board_and_model.py` and `make_custom_duel_env()` in `agent/src/battlesnake_ai/env/builder.py`).

## Training details

### Baseline loop (`train.py`)

- Builds a `SimpleCNN` from the observation channel count returned by `env.get_obs()`.
- Runs episodes with greedy `argmax` actions; illegal joint moves fall back to a random legal action.
- Writes console logs and TensorBoard scalars under `--log-dir` (default: `logs/`).

### DQN (`train_dqn.py`)

- Shared Q-network per snake row in the observation batch.
- Experience replay, target network sync, MSE TD loss, gradient clipping.
- ε-greedy per snake with correction when the greedy joint tuple is illegal.
- Checkpoints under `logs/checkpoints/` (`dqn_latest.pt`, optional `--checkpoint-every N`).
- Metrics: `dqn_metrics.jsonl` plus TensorBoard under `logs/tensorboard/`.

### Rainbow DQN (`train_rainbow.py`)

- Shared CNN backbone; dueling categorical (C51) head; prioritized replay; double Q; linear ε-decay.
- Checkpoints: `rainbow_latest.pt`.

### PPO (`train_ppo.py`)

- Shared backbone; actor–critic; on-policy rollouts with GAE and clipped surrogate.
- Checkpoints: `ppo_latest.pt`.

### Hybrid

- **Sequential:** `train_ppo_from_rainbow.py` copies Rainbow encoder weights into PPO (`--freeze-encoder` optional).
- **Ensemble (eval only):** `ensemble:rainbow.pt+ppo.pt` in `eval_agents.py` mixes Q-values and policy logits.

### Evaluation (`eval_agents.py`)

- Agent specs: `random`, `dqn:path.pt`, `rainbow:path.pt`, `ppo:path.pt`, `ensemble:a.pt+b.pt[:w_r:w_p]`.
- Writes `logs/eval_<timestamp>.json` with win/draw/loss tables.

Useful flags (DQN / Rainbow):

| Flag | Default | Purpose |
|------|---------|---------|
| `--num-players` | mode default | Snake count (standard modes only; duel stays 2) |
| `--replay-size` | 50000 | Replay buffer capacity |
| `--batch-size` | 64 | Minibatch size |
| `--gamma` | 0.99 | Discount factor |
| `--lr` | 1e-4 | Adam learning rate |
| `--train-after` | 500 | Min transitions before learning |
| `--target-update-every` | 500 | Steps between target net copies |
| `--epsilon-decay-steps` | 50000 | Linear ε decay schedule |
| `--gui` | off | Matplotlib board + training HUD (DQN / Rainbow / PPO) |
| `--gui-every` | 1 | Refresh GUI every N env steps when `--gui` is set |
| `--checkpoint-every` | 0 | Save policy every N episodes (DQN / Rainbow / PPO) |

View TensorBoard after a run:

```bash
tensorboard --logdir agent/logs/tensorboard
```

## Battlesnake Blackout submission

This repo includes a competition-ready server for [Battlesnake Blackout 2026](https://www.tnt.uni-hannover.de/bs-blackout-2026/) ([API docs](https://www.tnt.uni-hannover.de/bs-blackout-2026/doc)).

Blackout runs **restricted standard** games: 15×15 board, 4 snakes, **view radius 5**. Use a checkpoint trained with `--mode restricted_standard` (17 input channels). The Docker image ships:

`best_checkpoint/rainbow_v2_best.pt` (Rainbow v2: feature_dim 128, noisy nets, self-play)

Requires **hisss ≥ 1.3.0** (`requirements-server.txt`). With older hisss the server would catch a `BattleSnakeState` constructor error on every `/move` and silently return `FALLBACK_MOVE=up`, making the snake walk north into the wall.

### Endpoints

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/` | `{"author": "...", "color": "#RRGGBB"}` |
| `POST` | `/start` | `{}` (200 OK) |
| `POST` | `/move` | `{"move": "up"\|"down"\|"left"\|"right"}` |
| `POST` | `/end` | `{}` (200 OK) |

All handlers must respond in **under 500 ms**. The bundled Rainbow policy typically answers in ~1 ms on CPU.

### Run locally

```bash
cp .env.example .env   # set SNAKE_AUTHOR to your registered snake name
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=agent/src
export BATTLE_SNAKE_CHECKPOINT=best_checkpoint/rainbow_v2_best.pt

uvicorn server:app --host 0.0.0.0 --port 8000
```

Smoke-test against a real Blackout replay frame:

```bash
BATTLE_SNAKE_CHECKPOINT=best_checkpoint/rainbow_v2_best.pt \
  python scripts/test_blackout_api.py
```

### Deploy

**Docker** (bundles `best_checkpoint/`):

```bash
docker build -t battle-snake .
docker run -p 8000:8000 \
  -e SNAKE_AUTHOR="the sea snake" \
  -e SNAKE_COLOR="#4488ff" \
  battle-snake
```

After code changes, **rebuild and redeploy** the image — a running container will keep serving the old broken `/move` path until replaced.

**Railway / Render / Heroku**: use the included `Procfile` (`web: uvicorn server:app ...`). Set `SNAKE_AUTHOR`, `BATTLE_SNAKE_CHECKPOINT`, and expose port `8000`. Ensure the host installs from `requirements-server.txt` (hisss≥1.3.0).

Register on the Blackout site and submit your public HTTPS URL. The `author` field from `GET /` must match your registered snake name exactly.

## Implementation notes

- Importing `battlesnake_ai` applies a small patch for hisss view-radius row indexing (`env/hisss_view_radius_fix.py`).
- Observations from hisss are `(batch, width, height, channels)`; models permute to NCHW for `Conv2d`.
- Training logs under `agent/logs/` are generated artifacts; add them to `.gitignore` if you commit frequently.

## Example DQN training run

From `agent/` (after [setup](#setup)):

```bash
python scripts/train_dqn.py \
  --mode restricted_standard \
  --episodes 50 \
  --checkpoint-every 10 \
  --log-updates-every 50
```

**What you get**

| Path | Description |
|------|-------------|
| `logs/dqn_train_<timestamp>.log` | Console log (env steps at DEBUG, episodes and periodic updates at INFO) |
| `logs/dqn_metrics.jsonl` | One JSON object per gradient step (loss, ε, TD stats) |
| `logs/tensorboard/` | Scalars for episode length, per-snake reward, ε |
| `logs/checkpoints/dqn_latest.pt` | Policy + metadata written when training finishes |
| `logs/checkpoints/dqn_<timestamp>_ep{N}.pt` | Optional snapshots when `--checkpoint-every N` is set |

**Sample console output** (abbreviated):

```text
INFO - Building environment (restricted_standard)...
INFO - Observation shape: (4, 21, 21, 22) | channels=22
INFO - DQN training configuration: { "gamma": 0.99, "batch_size": 64, ... }
INFO - Episode 1 finished | env steps=142 | cumulative reward (per snake)=[0. 0. 1. 0.] | ε=0.9985 | replay size=142
INFO - DQN update | loss=0.0129 | TD mean=-0.0998 | ... | ε=0.9952 | batch=64 | ...
INFO - Saved checkpoint logs/checkpoints/dqn_20260522_193619_ep10.pt
INFO - Saved latest checkpoint logs/checkpoints/dqn_latest.pt
```

**Monitor while training**

```bash
tensorboard --logdir agent/logs/tensorboard
```

**Evaluate the checkpoint**

```bash
python scripts/eval_agents.py --mode restricted_standard --episodes 100 \
  --agents random dqn:logs/checkpoints/dqn_latest.pt
```

## Example Rainbow DQN training run (with GUI)

From `agent/` (after [setup](#setup)). Opens a Matplotlib window with the board and training HUD; use a smaller episode count for interactive runs.

```bash
python scripts/train_rainbow.py \
  --mode duel \
  --episodes 50 \
  --gui \
  --gui-every 2 \
  --checkpoint-every 10 \
  --log-updates-every 50
```

```bash
python scripts/train_rainbow.py \
  --mode duel \
  --episodes 50 \
  --gui \
  --gui-every 2 \
  --checkpoint-every 10 \
  --log-updates-every 50
```

`--gui-every 2` refreshes the board every two env steps (use `1` for every step; higher values reduce UI overhead on long runs).

**What you get**

| Path | Description |
|------|-------------|
| `logs/rainbow_train_<timestamp>.log` | Console log (`rainbow_train` logger name) |
| `logs/dqn_metrics.jsonl` | Gradient-step metrics (same JSONL schema as vanilla DQN) |
| `logs/tensorboard/` | Episode length, per-snake reward, ε |
| `logs/checkpoints/rainbow_latest.pt` | Policy + metadata at end of training |
| `logs/checkpoints/rainbow_<timestamp>_ep{N}.pt` | Optional snapshots with `--checkpoint-every N` |

**Sample console output** (abbreviated):

```text
INFO - hisss reward_cfg: {'living_reward': 0.0, 'terminal_reward': 1.0}
INFO - DQN training configuration: { "algorithm": "rainbow", "gamma": 0.99, "n_step": 3, ... }
INFO - Episode 1 finished | env steps=68 | cumulative reward (per snake)=[1. 0.] | ε=0.9986 | replay size=68
INFO - DQN update | loss=0.0412 | TD mean=0.0123 | ... | ε=0.9951 | batch=64 | ...
INFO - Saved checkpoint logs/checkpoints/rainbow_20260522_120000_ep10.pt
```

**Monitor while training**

```bash
tensorboard --logdir agent/logs/tensorboard
```

**Evaluate the checkpoint**

```bash
python scripts/eval_agents.py --mode duel --episodes 100 \
  --agents random rainbow:logs/checkpoints/rainbow_latest.pt
```

**Resume from a checkpoint** (GUI optional):

```bash
python scripts/train_rainbow.py --resume logs/checkpoints/rainbow_latest.pt \
  --mode duel --episodes 200 --gui
```

## License

Add a license file if you plan to publish or share the repo publicly.
