#!/bin/bash
#SBATCH --job-name=snake-ppo
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=p_12G
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-t2080ti-01
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

srun python3 agent/scripts/train_ppo.py \
    --mode restricted_standard \
    --episodes 20000 \
    --eval-every 50 \
    --eval-episodes 100 \
    --checkpoint-every 500 \
    --log-dir logs \
    --resume-checkpoint best_checkpoint/ppo_best.pt \
    --rainbow-opponent best_checkpoint/rainbow_v4_best_random.pt
