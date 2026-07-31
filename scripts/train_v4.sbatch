#!/bin/bash
#SBATCH --job-name=snake-v4
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=p_48G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --requeue

# Activate environment
source .venv/bin/activate

# Add agent src to PYTHONPATH
export PYTHONPATH="$(pwd)/agent/src"

# Rainbow DQN v4 — improved training with soft target updates, cosine LR,
# simplified reward shaping, larger replay buffer, and dual evaluation.
srun python3 agent/scripts/train_rainbow_v4.py \
    --mode restricted_standard \
    --episodes 20000 \
    --replay-size 500000 \
    --batch-size 512 \
    --feature-dim 128 \
    --noisy \
    --lr 6e-5 \
    --lr-min 1e-5 \
    --lr-restart-every 5000 \
    --n-step 3 \
    --soft-target-tau 0.005 \
    --train-every 4 \
    --train-after 2000 \
    --epsilon-start 0.3 \
    --epsilon-end 0.01 \
    --epsilon-decay-steps 100000 \
    --self-play \
    --self-play-fraction 0.6 \
    --pool-snapshot-every 200 \
    --tournament-every 500 \
    --quality-gate-games 20 \
    --quality-gate-min-wr 0.20 \
    --checkpoint-every 500 \
    --eval-every 50 \
    --eval-episodes 100 \
    --survival-shaping \
    --shaping-mode simplified \
    --living-bonus 0.005 \
    --kill-bonus 0.3 \
    --reset-optimizer \
    --log-updates-every 100 \
    --seed 42 \
    --log-dir logs \
    --resume best_checkpoint/rainbow_v2_best.pt
