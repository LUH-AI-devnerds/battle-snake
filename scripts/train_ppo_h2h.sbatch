#!/bin/bash
#SBATCH --job-name=snake-h2h
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=p_12G
#SBATCH --nodes=1
#SBATCH --exclude=gpu-p1080ti-01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue

# Continue from the deployed policy, targeting its one real weakness:
# head-to-head collisions were 57% of its deaths, and it lost them while
# being the longest snake on the board. Shaping now separates death causes
# and pays for winning an engine-attributed head-to-head.
#
# The deployed checkpoint itself joins the league, so the new policy has to
# beat its own predecessor rather than only the hand-written bots.

source .venv/bin/activate
export PYTHONPATH="$(pwd)/agent/src"

srun python3 agent/scripts/train_ppo.py \
    --mode restricted_standard \
    --num-players 4 \
    --survival-shaping \
    --h2h-death-penalty 0.35 \
    --h2h-win-bonus 0.7 \
    --episodes 6000 \
    --rollout-steps 8192 \
    --minibatch-size 512 \
    --lr 5e-5 --lr-end 1e-5 \
    --entropy-coef 0.004 --entropy-end 0.001 \
    --self-play-prob 0.25 \
    --league-checkpoints best_checkpoint/ppo_league_best.pt \
    --eval-every 250 --eval-episodes 50 \
    --benchmark-every 250 --benchmark-episodes 100 \
    --checkpoint-every 1000 \
    --log-dir logs \
    --resume-checkpoint best_checkpoint/ppo_league_best.pt
