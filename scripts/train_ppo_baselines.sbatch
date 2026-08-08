#!/bin/bash
#SBATCH --job-name=snake-base
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=p_12G
#SBATCH --nodes=1
#SBATCH --exclude=gpu-p1080ti-01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue

# Continue from the deployed policy against a more realistic league.
#
# The existing heuristic bots read env.get_state(), i.e. the whole board, which
# no real opponent can do. hungry_baseline is a faithful port of the official
# starter's HungryAgent and is limited to the fog-of-war view radius with food
# memory, so it behaves like the snakes we actually meet on the leaderboard.
# The deployed checkpoint is also in the league, so the new policy has to beat
# its own predecessor.

source .venv/bin/activate
export PYTHONPATH="$(pwd)/agent/src"

srun python3 agent/scripts/train_ppo.py \
    --mode restricted_standard \
    --num-players 4 \
    --survival-shaping \
    --episodes 8000 \
    --rollout-steps 8192 \
    --minibatch-size 512 \
    --lr 5e-5 --lr-end 1e-5 \
    --entropy-coef 0.004 --entropy-end 0.001 \
    --self-play-prob 0.20 \
    --baselines hungry_baseline hungry_baseline \
    --league flood_fill aggressive_hunter cautious \
    --league-checkpoints best_checkpoint/ppo_league_best.pt \
    --eval-every 250 --eval-episodes 50 \
    --benchmark-every 250 --benchmark-episodes 150 \
    --checkpoint-every 1000 \
    --log-dir logs \
    --resume-checkpoint best_checkpoint/ppo_league_best.pt
