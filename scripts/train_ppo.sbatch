#!/bin/bash
#SBATCH --job-name=snake-ppo
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=p_12G
#SBATCH --nodes=1
#SBATCH --exclude=gpu-p1080ti-01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --requeue

# Two-stage recipe:
#   1. Behaviour-clone the aggressive_hunter teacher (reaches bot level fast).
#   2. PPO fine-tune against the heuristic league + self-play, with the "best"
#      checkpoint gated on a fixed benchmark league so noise cannot promote a
#      weak policy.

# Activate environment
source .venv/bin/activate

# Add agent src to PYTHONPATH
export PYTHONPATH="$(pwd)/agent/src"

BC_CKPT=logs/checkpoints/ppo_bc_teacher.pt

srun python3 agent/scripts/pretrain_bc.py \
    --mode restricted_standard \
    --teacher aggressive_hunter \
    --iterations 60 \
    --steps-per-iter 8192 \
    --epsilon 0.25 \
    --eval-every 5 \
    --eval-episodes 50 \
    --log-dir logs \
    --out "$BC_CKPT"

srun python3 agent/scripts/train_ppo.py \
    --mode restricted_standard \
    --num-players 4 \
    --survival-shaping \
    --episodes 20000 \
    --rollout-steps 8192 \
    --minibatch-size 512 \
    --lr 5e-5 --lr-end 1e-5 \
    --entropy-coef 0.003 --entropy-end 0.001 \
    --self-play-prob 0.25 \
    --eval-every 250 --eval-episodes 50 \
    --benchmark-every 500 --benchmark-episodes 100 \
    --checkpoint-every 1000 \
    --log-dir logs \
    --resume-checkpoint "$BC_CKPT"
