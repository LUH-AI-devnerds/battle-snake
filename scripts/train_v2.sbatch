#!/bin/bash
#SBATCH --job-name=snake-rainbow-v2
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=p_80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue

# Activate environment
source .venv/bin/activate

# Add agent src to PYTHONPATH
export PYTHONPATH="$(pwd)/agent/src"

# Run training (resumed from OOM crash at ep~8605)
srun python3 agent/scripts/train_rainbow_v2.py \
    --mode restricted_standard \
    --episodes 15000 \
    --replay-size 200000 \
    --batch-size 256 \
    --feature-dim 128 \
    --noisy \
    --self-play \
    --pool-snapshot-every 50 \
    --checkpoint-every 500 \
    --eval-every 50 \
    --eval-episodes 100 \
    --survival-shaping \
    --survival-strategy aggressive \
    --lr 5e-5 \
    --n-step 5 \
    --log-updates-every 100 \
    --seed 42 \
    --log-dir logs \
    --resume logs/checkpoints/rainbow_v2_20260719_164731_ep8500.pt
