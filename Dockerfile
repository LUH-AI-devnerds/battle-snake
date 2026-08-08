FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-server.txt

COPY server.py ./
COPY agent/src/battlesnake_ai ./agent/src/battlesnake_ai
COPY best_checkpoint/ ./best_checkpoint/

ENV PYTHONPATH=/app/agent/src
# Railway reports the host's full core count while granting a tiny CPU quota;
# torch's default thread pool sizes itself to that count and thread contention
# then dominates a batch-of-one inference. Pin to 1 thread (also set in
# server.py, which is the belt to this braces since env vars must exist before
# the OpenMP/MKL runtime initializes).
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
# Patch hisss at image build time (site-packages is writable here; runtime-only patch
# can fail silently on read-only Railway layers and bring back FALLBACK_MOVE=up).
RUN python -c "from battlesnake_ai.env.hisss_view_radius_fix import apply_view_radius_row_index_fix; import sys; sys.exit(0 if apply_view_radius_row_index_fix() else 1)"

ENV BATTLE_SNAKE_CHECKPOINT="best_checkpoint/ppo_league_best.pt"
ENV SNAKE_AUTHOR="the sea snake"
ENV SNAKE_COLOR="#4488ff"
# Move selection: the league-trained policy ranks the moves and a one-step
# filter rejects suicide and losing head-to-heads.
#
# MOVE_STRATEGY=veto adds a lookahead search on top. Do not enable it until the
# search is fog-aware: it builds its board from the /move payload, which under
# fog of war truncates opponent bodies and hides food, so it treats occupied
# cells as free. Measured against the policy alone in the same games with
# production visibility, it is 13 points *worse* (33.0% vs 46.0%).
ENV MOVE_STRATEGY="model"
ENV SEARCH_BUDGET_MS="70"
# Model was trained with survival reward shaping; use raw Q at inference.
ENV SURVIVAL_FILTER="0"
ENV SURVIVAL_HUNGER_HEALTH="35"
ENV SURVIVAL_STRATEGY="aggressive"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
