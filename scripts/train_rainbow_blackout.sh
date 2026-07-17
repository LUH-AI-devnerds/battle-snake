#!/bin/bash
#SBATCH --job-name=rainbow-blackout
#SBATCH --partition=p_48G
#SBATCH --gres=gpu:a3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --output=/nfs/home/alastia/battle-snake/logs/slurm-%j.out
#SBATCH --error=/nfs/home/alastia/battle-snake/logs/slurm-%j.err

# Train Rainbow DQN for Battlesnake Blackout 2026 (restricted_standard).
#
# Uses a mid-tier GPU on purpose: training is CPU-bound (hisss env stepping).
# An A3090 is more than enough; no need for H100/L40S.
#
# Submit from anywhere:
#   sbatch /nfs/home/alastia/battle-snake/scripts/train_rainbow_blackout.sh
#
# One-time setup (Python 3.12+ required for hisss):
#   cd /nfs/home/alastia/battle-snake
#   uv venv .venv --python 3.12
#   source .venv/bin/activate
#   uv pip install -r agent/requirements.txt
#   #SBATCH --gres=gpu:l40s:1
#   #SBATCH --gres=gpu:p1080ti:1   # oldest option; fine for this tiny CNN

set -euo pipefail

REPO=/nfs/home/alastia/battle-snake
cd "${REPO}"

mkdir -p logs

if [[ -f "${REPO}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO}/.venv/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "Using active venv: ${VIRTUAL_ENV}"
else
  echo "ERROR: No virtualenv found at ${REPO}/.venv" >&2
  echo "Create it once (Python 3.12+ required for hisss):" >&2
  echo "  cd ${REPO}" >&2
  echo "  uv venv .venv --python 3.12" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  uv pip install -r agent/requirements.txt" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

cd "${REPO}/agent"

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST:-local}"
echo "Start: $(date -Is)"
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python scripts/train_rainbow.py \
  --resume "${REPO}/best_checkpoint/rainbow_20260715_214528_best.pt" \
  --mode restricted_standard \
  --episodes 5000 \
  --seed 42 \
  --epsilon-start 0.2 \
  --epsilon-end 0.05 \
  --epsilon-decay-steps 30000 \
  --beta-anneal-steps 30000 \
  --replay-size 80000 \
  --batch-size 128 \
  --train-after 1000 \
  --train-every 1 \
  --target-update-every 1000 \
  --max-grad-norm 10.0 \
  --n-step 3 \
  --num-atoms 51 \
  --v-min -1.0 --v-max 1.0 \
  --feature-dim 64 \
  --checkpoint-every 50 \
  --eval-every 50 \
  --eval-episodes 100 \
  --eval-seed 42 \
  --self-eval-every 250 \
  --self-eval-episodes 30 \
  --survival-shaping \
  --survival-strategy aggressive \
  --living-bonus 0.01 \
  --length-penalty 0.05 \
  --proximity-penalty 0.02 \
  --log-updates-every 100
# Resume workflow (training state now restored on --resume):
#   --resume logs/checkpoints/rainbow_<run_id>_best.pt
# This reloads model + optimizer + total_env_steps + best_win_rate, so epsilon/beta
# annealing continues from the saved step instead of restarting.
#
# Aggressive shaping: reward growth + closing on shorter prey; still penalize
# closing on equal/longer heads. Pair with SURVIVAL_STRATEGY=aggressive on deploy.

echo "End: $(date -Is)"
