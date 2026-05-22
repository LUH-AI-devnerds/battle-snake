# Battle Snake

Train reinforcement-learning agents for [Battlesnake](https://play.battlesnake.com/) using the [hisss](https://github.com/BattlesnakeOfficial/hisss) simulator. This repo includes a PyTorch training stack (CNN baseline and DQN), optional live board visualization, TensorBoard metrics, and a minimal FastAPI server stub for deploying a snake to the official API.

## Project layout

```
battle-snake/
├── agent/                    # Training package and scripts
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── train.py          # Greedy CNN rollout loop (no gradient updates)
│   │   └── train_dqn.py      # DQN with replay buffer and ε-greedy exploration
│   └── src/battlesnake_ai/
│       ├── env/              # hisss env factory + view-radius patch
│       ├── models/           # SimpleCNN, DQN
│       ├── training/         # Loops, replay buffer, logging
│       └── viz/              # Matplotlib training GUI
├── example_board_and_model.py  # Standalone hisss + dummy CNN demo
└── server.py                 # FastAPI `/` move endpoint stub
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
- Experience replay, target network sync, Huber loss, gradient clipping.
- ε-greedy per snake with correction when the greedy joint tuple is illegal.
- Metrics: `dqn_metrics.jsonl` plus TensorBoard under `logs/tensorboard/`.

Useful flags:

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
| `--gui` | off | Matplotlib board + training HUD |

View TensorBoard after a run:

```bash
tensorboard --logdir agent/logs/tensorboard
```

## Battlesnake server stub

`server.py` exposes a minimal [Battlesnake API](https://docs.battlesnake.com/api) shape for local testing:

```bash
uvicorn server:app --reload --port 8000
```

- `GET /` → `{"action": "move"}` (replace with model inference)
- `GET /info` → health check

Wire your trained policy into the root handler before playing on play.battlesnake.com.

## Implementation notes

- Importing `battlesnake_ai` applies a small patch for hisss view-radius row indexing (`env/hisss_view_radius_fix.py`).
- Observations from hisss are `(batch, width, height, channels)`; models permute to NCHW for `Conv2d`.
- Training logs under `agent/logs/` are generated artifacts; add them to `.gitignore` if you commit frequently.

## License

Add a license file if you plan to publish or share the repo publicly.
